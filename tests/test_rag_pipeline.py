import unittest

from app.core.rag_pipeline import build_prompt, generate_grounded_answer


class FakeLLMClient:
    def generate(self, model, prompt):
        return {"response": "Answer based on the context."}


class RAGPipelineTests(unittest.TestCase):
    def test_build_prompt_includes_question_and_context(self):
        prompt = build_prompt("What is the answer?", "Relevant document text")
        self.assertIn("What is the answer?", prompt)
        self.assertIn("Relevant document text", prompt)

    def test_generate_grounded_answer_uses_context(self):
        answer = generate_grounded_answer(
            question="What is the answer?",
            context="Relevant document text",
            llm_client=FakeLLMClient(),
            model_name="llama3",
        )

        self.assertIn("Answer based on the context", answer)


if __name__ == "__main__":
    unittest.main()
