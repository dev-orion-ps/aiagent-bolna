from ..knowledgebase import Knowledgebase
# (keep existing imports)

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
        
        response = await self.llm.get_chat_response(messages + [{"role": "system", "content": context}])
        return response

# (keep the rest of the file as is)
