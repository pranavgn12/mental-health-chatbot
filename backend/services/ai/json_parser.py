import json


class JSONParser:

    def parse(self, response):

        try:

            print("Raw response:")
            print(repr(response))

            result = json.loads(response)

            print("Parsed:", result)

            return result

        except Exception as e:

            print("JSON Parse Error:", e)
            print("Response was:")
            print(repr(response))

            return {

                "action": "CONTINUE",

                "title": None,

                "reason": "Invalid JSON returned by LLM.",

                "confidence": 0.0

            }