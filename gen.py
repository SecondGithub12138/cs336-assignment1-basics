
import torch
import json
from torch import Tensor
from tests.adapters import Tokenizer, TransformerLM

VOCAB_PATH = "outputs/TinyStoriesV2-GPT4-train/TinyStoriesV2-GPT4-train_vocab.pkl"
MERGES = "outputs/TinyStoriesV2-GPT4-train/TinyStoriesV2-GPT4-train_merges.pkl"

class Predictor:
    def __init__(self, model: TransformerLM, tokenizer: Tokenizer):
        # Initialize model
        self.model = model
        self.tokenizer = tokenizer

    # The input initially should be "str"
    # Then it should call Tokenizer encode() to get "list[int]" (strip it out) =========
    # Then it should call Transformer_LM forward to get predication
    # Picking up last token prediction, a logit
    # Apply softmax temperature scaling to it to get logit_v2
    # Top-p sampling(with a user-specified threshold value) to get the index, the token id
    # Tokenizer decode() it to get the bytes, further decode the bytes with utf-8 to get the string and return (strip it out) =========
    def predict(self, input: str) -> str:
        token_ids = self.tokenizer.encode(input) # list[int]=> Int[Tensor, " batch_size sequence_length"]
        # Truncate to context_length (256) if too long
        if len(token_ids) > 256:
            token_ids = token_ids[-256:]
        # 
        # test = torch.tensor([token_ids], dtype=torch.long)
        # print(f"test: is {test}")
        logit = self.model(torch.tensor([token_ids], dtype=torch.long))
        # print(f"logit: is {logit}")
        logit_after_softmax = self.softmax_temp_scal(logit[0, -1]) # fun: later can use logits = res[:, -1, :] for all & many batches maybe
        next_token_id = self.top_p(logit_after_softmax, p=0.9)
        return self.tokenizer.decode([next_token_id])

    def softmax_temp_scal(self, x: torch.Tensor, temperature: float = 0.7):
        # print(f"x: is {x}")
        x_max = torch.max(x, dim=-1, keepdim=True).values
        # print(f"x_max: is {x_max}")
        x_stable = x - x_max
        # print(f"x_stable: is {x_stable}")
        x_exp = torch.exp(x_stable / temperature)
        # print(f"x_exp: is {x_exp}")
        res =  x_exp / torch.sum(x_exp, dim=-1, keepdim=True)
        # print(f"res: is {res}")
        return res

    def top_p(self, probs: torch.Tensor, p: float = 0.9) -> int:
        sorted_probs, sorted_idxs = torch.sort(probs, dim=-1, descending=True)
        sorted_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = sorted_probs <= p
        mask[0] = True
        sorted_probs = sorted_probs[mask]
        sorted_probs = sorted_probs/torch.sum(sorted_probs, dim=-1)
        sorted_probs_idx = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_idxs[sorted_probs_idx].item()
        """
        Top-p (nucleus) sampling.

        Args:
            probs: 概率分布，shape (vocab_size,)，已经过 softmax
            p: 累积概率阈值，默认 0.9

        Returns:
            采样得到的 token id
        """
        # 1. 按概率从大到小排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # 2. 计算累积概率
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

        # 3. 找到累积概率刚好超过 p 的位置
        # 创建 mask: 保留累积概率 <= p 的 token
        # 注意：我们至少要保留第一个 token
        mask = cumsum_probs <= p
        # 确保至少有一个 token（防止 p 太小）
        mask[0] = True

        # 4. 过滤出符合条件的概率和索引
        filtered_probs = sorted_probs[mask]
        filtered_indices = sorted_indices[mask]

        # 5. 重新归一化概率
        filtered_probs = filtered_probs / filtered_probs.sum()

        # 6. 从过滤后的分布中采样
        # multinomial 返回采样的索引（在 filtered_indices 中的位置）
        sampled_idx = torch.multinomial(filtered_probs, num_samples=1)

        # 7. 映射回原始 vocab 中的 token id
        token_id = filtered_indices[sampled_idx].item()

        return token_id

    def predict_all(self, input: str, max_gen: int):
        next = None
        cnt = 0
        curInput = input
        print(f"curInput: {curInput}")
        print("reply: ")
        while next not in self.tokenizer.special_tokens and cnt < max_gen:
            next = self.predict(curInput)
            print(next, end="")
            curInput = curInput + next
            print("")
            print(f"curInput: {curInput}")
            print(f"============================")
            cnt+=1
        print("")
        return curInput

def main():
    # print("do something")
    # run as a service 
    # todo, load checkpoint
    with open("config_default.json", 'r') as f:
        config = json.load(f)
    model = TransformerLM(config["vocab_size"], config["context_length"], config["d_model"], config["num_layers"], config["num_heads"], config["d_ff"], config["rope_theta"], None)
    
    with open("checkpoints/step_200000", "rb") as f:
        checkpoint = torch.load(f)
    model.load_state_dict(checkpoint["model_state_dict"])

    tok = Tokenizer.from_files(VOCAB_PATH, MERGES, ["<|endoftext|>"])
    
    predictor = Predictor(model, tok)
    
    predictor.predict_all("I am the king", 5000)


if __name__ == "__main__":
    main()
