from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import List
import os
from fastapi.responses import StreamingResponse
import asyncio
import json, time
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchParameter(BaseModel):
    search_parameter: str
    context: str


@app.post("/get_text")
async def get_text(search_parameter: SearchParameter):
    start_time = time.time()
    async def generate_chunks():
        # Call the ChatGroq API
        
        chat = ChatGroq(
            temperature=0,
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-8b-8192",
            streaming=True,
        )
        system = """
        You are  an experienced legal research assistant, you are tasked with analyzing 
        how the provided "context" is related to the search parameter "search_parameter" 
        Your task is to write  very concise 3 bullets outlining how the "context" is related to the search parameter, so that your boss can easily understand if "context" is worth spending time. Include facts, names, dates, and other relevant information, in your answer, so that your boss can have most useful facts, and reasons at once. Keep the reasons very concise, and to the point, so that your boss can quickly understand the relevance of the context to the search parameter, but it should be informative enough to give a clear picture of the context.

        """
        human = f"The search parameter is {search_parameter.search_parameter} and the context is {search_parameter.context} Analyze carefully, and write how the context is related to the search parameter"
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

        chain = prompt | chat
        for chunk in chain.stream(
            {
                "search_parameter": search_parameter.search_parameter,
                "context": search_parameter.context
                }
            ): 
            yield chunk.content
    end_time = time.time()
    print(f"Time taken: {end_time-start_time}")
    return StreamingResponse(generate_chunks())


# @app.post("/analyze", response_class=StreamingResponse)
# async def analyze(search_parameter: SearchParameter):
#     start_time = time.time()
#     # Call the ChatGroq API
#     chat = ChatGroq(
#         temperature=0,
#         groq_api_key=GROQ_API_KEY,
#         model_name="llama3-8b-8192",
#         streaming=True,
#     )
#     system = """
# You are  an experienced legal research assistant, you are tasked with analyzing 
# how the provided "context" is related to the search parameter "search_parameter" 
# Your task is to write  very concise 3 bullets outlining how the "context" is related to the search parameter, so that your boss can easily understand if "context" is worth spending time. Include facts, names, dates, and other relevant information, in your answer, so that your boss can have most useful facts, and reasons at once. Keep the reasons very concise, and to the point, so that your boss can quickly understand the relevance of the context to the search parameter, but it should be informative enough to give a clear picture of the context.

#     """
#     human = f"The search parameter is {search_parameter.search_parameter} and the context is {search_parameter.context} Analyze carefully, and write how the context is related to the search parameter"
#     prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

#     chain = prompt | chat
#     chunks = ""


#     for chunk in chain.stream(
#         {
#             "search_parameter": search_parameter.search_parameter,
#             "context": search_parameter.context,
#         }
#     ):
#         chunks += (jsonable_encoder(chunk.content))

#     end_time = time.time()
#     print(f"Time taken: {end_time-start_time}")
#     print(Response(content=json.dumps(chunks), media_type="application/json"))


if "__name__" == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost",port=8030)