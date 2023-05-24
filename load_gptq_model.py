from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

from auto_gptq import AutoGPTQForCausalLM

MODEL_DIR = "../models/Wizard-Vicuna-7B-Uncensored-GPTQ"
MODEL_BASENAME = "Wizard-Vicuna-7B-Uncensored-GPTQ-4bit-128g.no-act-order"

from transformers import pipeline
from transformers import AutoTokenizer

if __name__ == '__main__':
    device = "cuda:0"
    model = AutoGPTQForCausalLM.from_quantized(MODEL_DIR, model_basename=MODEL_BASENAME,
                                               device=device, use_triton=False,
                                               use_safetensors=True, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    pipeline = pipeline(task='text-generation', model=model, tokenizer=tokenizer, max_length=500)

    template = """Question: {question}
        Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm = HuggingFacePipeline(pipeline=pipeline)

    # create simple chain
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # set and run for question
    question = "What is difference between desert and sea?"
    answer = llm_chain.run(question)
    print(answer)
