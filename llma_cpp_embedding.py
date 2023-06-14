from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains.question_answering import load_qa_chain

from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import FAISS

import pickle

MODEL_PATH = "./models/Wizard-Vicuna-7B-Uncensored.ggmlv3.q4_0.bin"

if __name__ == '__main__':
    llama_embedding = LlamaCppEmbeddings(model_path=MODEL_PATH)

    text = ["The name hololive was initially used for COVER's 3D stream distribution app, launched in December 2017, and later its female VTuber agency, whose first generation debuted from May to June 2018. In December 2019, this hololive branch was merged with COVER's male HOLOSTARS agency and INoNaKa (INNK) Music label to form a unified 'hololive production' brand. In 2019 and 2020, the agency debuted three overseas branches: hololive China (since disbanded), hololive Indonesia, and hololive English.", "This tex is just another text as place holder. It is for checking whether embedding is working"]
    vectorstore= FAISS.from_texts(text, embedding=llama_embedding)
    store_name = "faiss_emb_store"
    with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Verbose is required to pass to the callback manager
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=MODEL_PATH, callback_manager=callback_manager, verbose=True
    )

    query = "Which branches deos hololive has?"

    # perform search in vector store
    docs = vectorstore.similarity_search(query=query, k=1)
    print("Found docs:")
    print(docs)

    # with callback_manager as cb:
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    # response = chain.run(input_documents=docs, question=query)
    response = chain.run(input_documents=docs, question=query)
    print("\nResult:")
    print(response)
