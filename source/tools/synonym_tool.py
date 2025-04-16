
import requests
import BeautifulSoup
from smolagents import tool

@tool
def get_synonym(word: str) -> str:
    """
    Retrieves synonyms online for a given word from an API.
    The agent can use this tool get inspierd to use new term. 

    Args:
        word (str): The word for which synonyms are needed.

    Returns:
        list: list of synonyms, or an error message if none are found.
    """
    try:
        url = f'https://www.thesaurus.com/browse/{word}'
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            return f"Error: Unable to fetch data (status code {response.status_code})."

        soup = BeautifulSoup(response.text, 'lxml')
        section = soup.find('section', {'data-type': 'synonym-antonym-module'})

        if not section:
            return "No synonyms found."

        strong_match = section.find("ul").find_all("li")

        if not strong_match:
            return "No synonyms available."

        synonyms = [i.text.strip() for i in strong_match]
        return synonyms

    except requests.exceptions.RequestException as e:
        return f"Error: Unable to connect to Thesaurus.com ({str(e)})."
    except Exception as e:
        return f"Unexpected error: {str(e)}."