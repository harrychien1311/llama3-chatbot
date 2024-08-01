from tokenizer import Tokenizer
from logging import getLogger
import torch
from typing import List, Optional, Tuple, TypedDict
import argparse

logger = getLogger(__name__)
        
def generate(
        model,
        tokenizer,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            model: The pretrained LLama model
            tokenizer: The tokenizer of the pretrained LLama model
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        bsz = len(prompt_tokens)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        total_len = max_gen_len + max_prompt_len

        pad_id = tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)

        eos_reached = torch.tensor([False] * bsz)
        input_text_mask = tokens != pad_id

        stop_tokens = torch.tensor(list(tokenizer.stop_tokens))

        for cur_pos in range(min_prompt_len, total_len):
            logits = model(tokens[:, :cur_pos])
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )

            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            # cut to after eos tok if any
            for stop_token in tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                except ValueError:
                    pass
            out_tokens.append(toks)
        return out_tokens

def text_completion(
    model,
    tokenizer,
    prompts: List[str],
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
    echo: bool = False,
):
    """
    Perform text completion for a list of prompts using the language generation model.

    Args:
        model (torch.jit.trace): pretrained LLama model
        tokenizer: tokenizer of the  LLama model
        prompts (List[str]): List of text prompts for completion.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
            If not provided, it's set to the model's maximum sequence length minus 1.
        echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

    Returns:
        List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

    Note:
        This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
        If logprobs is True, token log probabilities are computed for each generated token.

    """
    assert max_gen_len is not None
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    generation_tokens = generate(
        model,
        tokenizer,
        prompt_tokens=prompt_tokens,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
    )
    return [{"generation": tokenizer.decode(t)} for t in generation_tokens]

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def main(args):
    tokenizer = Tokenizer(f"weights/{args.tokenizer}")
    # Load traced model
    model = torch.jit.load("weights/llama3_scripted.pt")
    model.eval()
    example_texts = ["Messi is",
                     "Who is the president of France?",
                     "Hoang Sa and Truong Sa belong to"]
    generated_texts = text_completion(model, tokenizer, example_texts, max_gen_len=args.max_gen_len, temperature=args.temperature)
    for prompt, result in zip(example_texts, generated_texts):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default= "tokenizer.model", help="pretrained tokenizer file")
    parser.add_argument("--max_gen_len", type=int, default = 10, help="Max length of text generated")
    parser.add_argument("--temperature", type = float, default=-1, help="Temperature value for controlling randomness in sampling")
    args = parser.parse_args()
    main(args)
    