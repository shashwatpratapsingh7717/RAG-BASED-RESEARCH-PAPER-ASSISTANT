from groq import Groq
class GroqLLM:
    def __init__(
        self,
        api_key: str,
        model_name: str
    ):

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Add it to your .env file."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model_name = model_name
    def generate(
        self,
        question: str,
        context: str,
        instruction: str = ""
    ) -> str:
        system_prompt = """
You are an AI research-paper assistant.
You must follow these rules:
1. Answer only from the supplied context.
2. Never invent facts, numbers, papers, authors, or citations.
3. If the answer is not supported by the context,
   clearly say that the information was not found
   in the uploaded papers.
4. Be concise but technically accurate.
5. Use headings and bullet points when useful.
""".strip()

        if instruction:
            system_prompt += (
                "\n\nAdditional instruction:\n"
                + instruction
            )
        user_prompt = f"""
Question:
{question}
Retrieved Context:
{context}
Answer:
""".strip()
        response = (
            self.client
            .chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],

                temperature=0.1
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )