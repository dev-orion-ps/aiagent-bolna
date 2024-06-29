import json
import os
from dotenv import load_dotenv
from .base_agent import BaseAgent
from bolna.helpers.utils import format_messages
from bolna.llms import OpenAiLLM
from bolna.prompts import CHECK_FOR_COMPLETION_PROMPT
from bolna.helpers.logger_config import configure_logger
from ..knowledgebase import Knowledgebase

load_dotenv()
logger = configure_logger(__name__)

class ContextualConversationalAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.knowledgebase = Knowledgebase()

    def add_to_knowledgebase(self, document):
        self.knowledgebase.add_document(document)

    async def get_response(self, input_text):
        relevant_info = self.knowledgebase.query(input_text)
        context = "Relevant information: " + " ".join(relevant_info)
        
        # (keep existing logic, but add context to the prompt)
        messages = self.history + [{"role": "user", "content": input_text}]
        response = await self.llm.get_chat_response(messages + [{"role": "system", "content": context}])
        return response

class StreamingContextualAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self.conversation_completion_llm = OpenAiLLM(classification_model=os.getenv('CHECK_FOR_COMPLETION_LLM', llm.classification_model))
        self.history = [{'content': ""}]

    async def check_for_completion(self, messages, check_for_completion_prompt = CHECK_FOR_COMPLETION_PROMPT):
        prompt = [
            {'role': 'system', 'content': check_for_completion_prompt},
            {'role': 'user', 'content': format_messages(messages, use_system_prompt=True)}]
        answer = None
        response = await self.conversation_completion_llm.generate(prompt, True, False, request_json=True)
        answer = json.loads(response)
        logger.info('Agent: {}'.format(answer['answer']))
        return answer

    async def generate(self, history, synthesize=False):
        async for token in self.llm.generate_stream(history, synthesize=synthesize):
            logger.info('Agent: {}'.format(token))
            yield token

# (keep the rest of the file as is)
