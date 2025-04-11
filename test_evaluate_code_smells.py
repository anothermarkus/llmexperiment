import unittest

from code_smell_utils import extract_smells_from_response

class TestSmellExtraction(unittest.TestCase):
    def test_async_and_exception_detection(self):
        input_text = (
            "Issue:\n- The `.Result` property is used to get the result of the `GetAsync` method. "
            "This can lead to deadlocks in asynchronous code. It is recommended to use the `await` keyword instead.\n\n"
            "- catch (Exception ex);\n+ catch (HttpRequestException ex);\n\n"
            "Issue:\n- The `Exception` class is too broad... catch more specific exceptions like `HttpRequestException`."
        )
        expected = {"async", "exception"}
        actual = set(extract_smells_from_response(input_text, "DotNet"))
        self.assertEqual(expected, actual)

if __name__ == "__main__":
    unittest.main()
