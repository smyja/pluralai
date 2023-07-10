import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from llama_index import download_loader, StorageContext, load_index_from_storage
from llama_index.indices.postprocessor import SentenceEmbeddingOptimizer
from llama_index.output_parsers import GuardrailsOutputParser
from llama_index.llm_predictor import StructuredLLMPredictor
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
    DEFAULT_REFINE_PROMPT_TMPL,
)
from pydantic import BaseModel,Field

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
def query_data(request: QueryRequest):
    llm_predictor = StructuredLLMPredictor()
    JSONReader = download_loader("JSONReader")
    loader = JSONReader()
    # documents = loader.load_data(Path('./scraped_data.json'))
    # index = VectorStoreIndex.from_documents(documents)

    # index.storage_context.persist()
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    rail_spec = """
        <rail version="0.1">

        <output>
            <object name="documentation" format="length: 2">
                <string
                    name="answer"
                    description="an answer to the question asked with a formatted code block if applicable"
                />
                <url
                    name="follow_up_url"
                    description="A source link or reference url where I can read more about this"
                    required="true"
                    format="valid-url"
                    on-fail-valid-url="filter"
                />
            </object>
        </output>

        <prompt>

        Query string here.

        @xml_prefix_prompt

        {output_schema}

        @json_suffix_prompt_v2_wo_none
        </prompt>
        </rail>
        """
    output_parser = GuardrailsOutputParser.from_rail_string(rail_spec, llm=llm_predictor.llm)

    fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
    fmt_refine_tmpl = output_parser.format(DEFAULT_REFINE_PROMPT_TMPL)
    qa_prompt = QuestionAnswerPrompt(fmt_qa_tmpl, output_parser=output_parser)
    refine_prompt = RefinePrompt(fmt_refine_tmpl, output_parser=output_parser)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(node_postprocessors=[SentenceEmbeddingOptimizer(percentile_cutoff=0.5)],    
    text_qa_template=qa_prompt,
    refine_template=refine_prompt,
    llm_predictor=llm_predictor,
    response_mode="compact",
    similarity_cutoff=0.7)
    response = query_engine.query(request.question)
    

    if not response:
        raise HTTPException(status_code=404, detail="No results found")

    return QueryResponse(answer=str(response))
