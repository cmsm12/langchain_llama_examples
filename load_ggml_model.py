from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

MODEL_PATH = "./wizardLM-7B.ggml.q4_0.bin"

if __name__ == '__main__':

    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # load model
    llm = LlamaCpp(
        model_path=MODEL_PATH, callback_manager=callback_manager, verbose=True
    )

    # create simple chain
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # set and run for question
    question = "What is difference between desert and sea?"
    answer = llm_chain.run(question)
    print(answer)
