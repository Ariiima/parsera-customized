import json
from parsera import Parsera
from dotenv import load_dotenv

load_dotenv()


def crawl_products(url: str):
    try:
        elements = {
            "name": "extract the product name",
            "price": "extract the product price in number",
            "description": "extract the product description",
            "image": "extract the product image URL",
            "link": "extract the product page URL",
            "tags": "extract the list of tags for products example:[small, colorful, t-shirt, etc.]"
        }
        scrapper = Parsera()
        try:
            result = scrapper.run(url=url, elements=elements)
            if not result:
                return {"error": "No data extracted"}
            return result
        except Exception as e:
            return {"error": f"Extraction failed: {str(e)}"}

    except Exception as e:
        print(f"Error initializing scraper: {e}")
        return {"error": f"Initialization failed: {str(e)}"}


result = crawl_products("https://gianflower.com/product/%d8%a7%d8%b1%da%a9%db%8c%d8%af%d9%87-%d9%85%db%8c%d9%86%db%8c%d8%a7%d8%aa%d9%88%d8%b1%db%8c/")
print(json.dumps(result, indent=4))
