from logger import logger

def query_chain(chain, user_input: str):
    try:
        logger.debug(f"Running chain for input: {user_input}")

        result = chain.invoke({"input": user_input})

        response = {
            "response": result["answer"],
            "sources": [
                doc.metadata.get("source", "")  
                for doc in result["context"]
            ]
        }

        logger.debug(f"Chain response: {response}")
        return response

    except Exception as e:
        logger.exception("Error in query chain")
        raise
