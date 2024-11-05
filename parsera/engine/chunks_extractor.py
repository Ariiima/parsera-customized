import json
import math
from typing import Callable

import tiktoken
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markdownify import MarkdownConverter

from parsera.engine.simple_extractor import TabularExtractor

SYSTEM_MERGE_PROMPT_TEMPLATE = """
Your goal is to merge data extracted from different parts of the page into one json.
Keep the original structure, but make sure to correctly merge data to not have duplicates, prefer rows with more data.

In case of conflicting values, take the data that is further from the boarder between the files.
## Example: merging conflicting values
```
[
    {"name": "zero element", "price": "123"},
    {"name": "first element", "price": "100"},
    {"name": "second element", "price": "200"},
    {"name": "third element", "price": "999"},
]
```

and 

```
[
    {"name": "second element", "price": "123"},
    {"name": "third element", "price": "400"},
]
```

In the case above the "second element"'s price should be taken from the first json, cause it's further from the boarder
between the files, while "third element"'s price should be taken from the second json. The final output should be:

Merged json:
```
[
    {"name": "zero element", "price": "123"},
    {"name": "first element", "price": "100"},
    {"name": "second element", "price": "200"},
    {"name": "third element", "price": "400"},
]
```

## Example: merging missing values
```
[
    {"name": "zero element", "price": "123"},
    {"name": "first element", "price": "100"},
    {"name": "second element", "price": "200"},
    {"name": "third", "price": null},
]
```

and 

```
[
    {"name": "second element", "price": "123"},
    {"name": "third element", "price": "400"},
    {"name": "fourth element", "price": "350"},
]
```

In this case the "third element" should be taken from the second json, because it contains missing value for the "price"
and fixed truncated name "third". The final output should be:

Merged json:
```
[
    {"name": "zero element", "price": "123"},
    {"name": "first element", "price": "100"},
    {"name": "second element", "price": "200"},
    {"name": "third element", "price": "400"},
    {"name": "fourth element", "price": "350"},
]
```

"""

EXTRACTOR_MERGE_PROMPT_TEMPLATE = """
Elements requested by user:
```
{elements}
```

All jsons from different parts of the page:
{jsons_list}

Merged json:
"""

APPEND_TABULAR_EXTRACTOR_SYSTEM_PROMPT = """
Your goal is to continue the sequence extracted from the previous chunk of the page by adding elements from the new page
chunk. For each extracted field, you must extract the EXACT CSS selector from the HTML content that was used to find that element.
DO NOT make up or guess selectors - only use what's present in the HTML.

Important rules for selectors:
1. Use the most specific and accurate selector from the actual HTML
2. Include all necessary parent elements to uniquely identify the element
3. If you can't find a specific selector in the HTML, use null or empty string
4. Include class names, IDs, and other attributes exactly as they appear in the HTML
5. Pay attention to the actual HTML structure in the content

## Example: continue truncated json with selectors
```json
[
    {
        "name": {
            "value": "Product Name",
            "selector": "div.product__title > h1.product-single__title"
        },
        "price": {
            "value": "25.00",
            "selector": "span.price-item.price-item--regular"
        },
        "description": {
            "value": "Product description text",
            "selector": "div.product-single__description.rte"
        }
    }
]
```

Only extract selectors that you can verify in the provided HTML content.
If you cannot find a proper selector, use null rather than making one up.
"""

APPEND_EXTRACTOR_PROMPT_TEMPLATE = """
Fill missing values, fix truncated values and continue this sequence:
```
{previous_data}
```

The current truncated page chunk:
---
{markdown}
---

You are looking for the following elements from the truncated page chunk:
```
{elements}
```

For each element, also identify and return the CSS selector that can be used to locate it.
Output json with fixed previous sequence, new rows, and CSS selectors:
"""


class ChunksTabularExtractor(TabularExtractor):
    system_merge_prompt = SYSTEM_MERGE_PROMPT_TEMPLATE
    prompt_merge_template = EXTRACTOR_MERGE_PROMPT_TEMPLATE
    append_system_prompt = APPEND_TABULAR_EXTRACTOR_SYSTEM_PROMPT
    append_prompt_template = APPEND_EXTRACTOR_PROMPT_TEMPLATE
    overlap_factor = 3

    def __init__(
        self,
        model: BaseChatModel,
        chunk_size: int = 100000,
        token_counter: Callable[[str], int] | None = None,
        converter: MarkdownConverter | None = None,
    ):
        """Initialize ChunksTabularExtractor

        Args:
            model (BaseChatModel | None, optional): LangChain Chat Model. Defaults to None which
            invokes usage of GPT4oMiniModel.
            chunk_size (int, optional): Number of tokens per chunk, should be below context size of
            the model used. Defaults to 100000.
            token_counter (Callable[[str], int] | None, optional): Function used to estimate number
                of tokens in chunks. If None will use OpenAI tokenizer for gpt-4o model.
                Defaults to None.
            converter (MarkdownConverter | None, optional): converter of HTML, before it goes to
                the model. Defaults to None.
        """
        super().__init__(
            model=model,
            converter=converter,
        )
        if token_counter is None:

            def count_tokens(text):
                # Initialize the tokenizer for GPT-4o-mini
                encoding = tiktoken.get_encoding("o200k_base")

                # Count tokens
                tokens = encoding.encode(text)
                return len(tokens)

            token_counter = count_tokens

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // self.overlap_factor,
            length_function=token_counter,
        )
        self.chunks_data = None

    async def extract(
        self,
        markdown: str,
        attributes: dict[str, str],
        previous_data: list[dict] | None = None,
    ) -> list[dict]:
        elements = json.dumps(attributes)
        if not previous_data:
            human_msg = self.prompt_template.format(
                markdown=markdown, elements=elements
            )
        else:
            # Convert previous_data to list if it's not already
            if not isinstance(previous_data, list):
                previous_data = list(previous_data)
            
            # Ensure we have valid data to slice
            if len(previous_data) > 0:
                cutoff = max(1, math.ceil(len(previous_data) / self.overlap_factor))
                previous_tail = json.dumps(previous_data[-cutoff:])
            else:
                previous_tail = "[]"
            
            human_msg = self.append_prompt_template.format(
                markdown=markdown, elements=elements, previous_data=previous_tail
            )
        
        messages = [
            SystemMessage(self.system_prompt),
            HumanMessage(human_msg),
        ]
        output = await self.model.ainvoke(messages)
        parser = JsonOutputParser()
        output_dict = parser.parse(output.content)
        return output_dict

    async def merge_all_data(
        self, all_data: list[list[dict]], attributes: dict[str, str]
    ) -> dict:
        elements = json.dumps(attributes)
        json_list = ""
        for data in all_data:
            json_list += "``` \n" + json.dumps(data) + "\n ```\n"

        human_msg = self.prompt_merge_template.format(
            elements=elements, jsons_list=json_list
        )
        messages = [
            SystemMessage(self.system_merge_prompt),
            HumanMessage(human_msg),
        ]
        output = await self.model.ainvoke(messages)
        parser = JsonOutputParser()
        output_dict = parser.parse(output.content)
        return output_dict

    async def run(
        self,
        content: str,
        attributes: dict[str, str],
    ) -> dict:
        if self.system_prompt is None:
            raise ValueError("system_prompt is not defined for this extractor")
        if self.prompt_template is None:
            raise ValueError("prompt_template is not defined for this extractor")

        chunks = self.text_splitter.create_documents([content])
        
        # Print token count for each chunk using the token_counter from initialization
        for i, chunk in enumerate(chunks):
            token_count = self.text_splitter._length_function(chunk.page_content)
            print(f"Chunk {i + 1} token count: {token_count}")

        if len(chunks) > 1:
            self.chunks_data = []
            chunk_data = None
            for _, element in enumerate(chunks):
                chunk_data = await self.extract(
                    markdown=element, previous_data=chunk_data, attributes=attributes
                )
                self.chunks_data.append(chunk_data)

            output_dict = await self.merge_all_data(
                all_data=self.chunks_data, attributes=attributes
            )
        else:
            output_dict = await self.extract(markdown=chunks[0], attributes=attributes)

        return output_dict
