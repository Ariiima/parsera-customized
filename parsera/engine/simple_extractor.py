import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from markdownify import MarkdownConverter

SIMPLE_EXTRACTOR_PROMPT_TEMPLATE = """
Having following page content:
```
{markdown}
```

Return the following elements from the page content:
```
{elements}
```

Output json:
"""


class Extractor:
    system_prompt = None
    prompt_template = SIMPLE_EXTRACTOR_PROMPT_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel,
        converter: MarkdownConverter | None = None,
    ):
        self.model = model
        if converter is None:
            self.converter = MarkdownConverter()
        else:
            self.converter = converter

    async def run(self, content: str, attributes: dict[str, str]) -> list[dict]:
        if self.system_prompt is None:
            raise ValueError("system_prompt is not defined for this extractor")
        if self.prompt_template is None:
            raise ValueError("prompt_template is not defined for this extractor")

        elements = json.dumps(attributes)
        human_msg = self.prompt_template.format(markdown=content, elements=elements)
        messages = [
            SystemMessage(self.system_prompt),
            HumanMessage(human_msg),
        ]
        output = await self.model.ainvoke(messages)
        parser = JsonOutputParser()
        output_dict = parser.parse(output.content)

        return output_dict



TABULAR_EXTRACTOR_SYSTEM_PROMPT = """
Your goal is to find the elements from the webpage content and return them in json format.
For example if user asks:
Return the following elements from the page content:
```
{
    "name": "name of the listing",
    "price": "price of the listing"
}
```
Make sure to return json with the list of corresponding values and their CSS selectors.
Output json:
```json
[
{
"name": {"value": "name1", "selector": "#item-1 .name"},
"price": {"value": "100", "selector": "#item-1 .price"}
},
{
"name": {"value": "name2", "selector": "#item-2 .name"},
"price": {"value": "150", "selector": "#item-2 .price"}
},
{
"name": {"value": "name3", "selector": "#item-3 .name"},
"price": {"value": "300", "selector": "#item-3 .price"}
}
]
```

If users asks for a single field:
Return the following elements from the page content:
```
{
    "link": "link to the listing",
}
```
Make sure to return json with only this field and its selector
Output json:
```json
[
   {"link": {"value": "https://example.com/link1", "selector": ".item-1 a"}},
{"link": {"value": "https://example.com/link2", "selector": ".item-2 a"}},
{"link": {"value": "https://example.com/link3", "selector": ".item-3 a"}}
]
```

"""



class TabularExtractor(Extractor):
    system_prompt = TABULAR_EXTRACTOR_SYSTEM_PROMPT


LIST_EXTRACTOR_SYSTEM_PROMPT = """
Your goal is to find the elements from the webpage content and return them in json format.
For example if user asks:
Return the following elements from the page content:
```
{
    "name": "name of the listing",
    "price": "price of the listing"
}
```
Make sure to return json with the list of corresponding values.
Output json:
```json
{
    "name": ["name1", "name2", "name3"],
    "price": ["100", "150", "300"]
}
```

If users asks for a single field:
Return the following elements from the page content:
```
{
    "link": "link to the listing",
}
```
Make sure to return json with only this field
Output json:
```json
{
    "link": ["https://example.com/link1", "https://example.com/link2", "https://example.com/link3"]
}
```

If no data is found return empty json:
```json
[]
```

"""


class ListExtractor(Extractor):
    system_prompt = LIST_EXTRACTOR_SYSTEM_PROMPT


ITEM_EXTRACTOR_SYSTEM_PROMPT = """
Your goal is to find the elements from the webpage content and return them in json format.
For example if user asks:
Return the following elements from the page content:
```
{
    "name": "name of the listing",
    "price": "price of the listing"
}
```
Make sure to return a json with corresponding values.
Output json:
```json
{
    "name": "name1",
    "price": "100"
}
```

If users asks for a single field:
Return the following elements from the page content:
```
{
    "link": "link to the listing",
}
```
Make sure to return json with only this field
Output json:
```json
{
    "link": "https://example.com/link1"]
}
```

If no data is found return empty json:
```json
[]
```

"""


class ItemExtractor(Extractor):
    system_prompt = ITEM_EXTRACTOR_SYSTEM_PROMPT
