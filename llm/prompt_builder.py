class PromptBuilder:
    @staticmethod
    def build_tumor_report_prompt(prediction_results):
        prompt = f"Write a brief medical report based on these tumor detection results: {prediction_results}"
        return prompt
