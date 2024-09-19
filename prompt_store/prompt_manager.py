import yaml
import os

class PromptManager:
    def __init__(self):
        """
        Initialize the PromptManager with the YAML file.
        """
        file_dir = os.path.dirname(__file__)
        self.yaml_file = os.path.abspath(os.path.join(file_dir, "..", "config", "prompts.yaml"))
        self.prompts = self.load_prompts()

    def load_prompts(self):
        """
        Load the prompts from the YAML file.

        Returns:
            dict: The dictionary containing prompts loaded from the file. If the file
                  does not exist, it returns a dictionary with an empty "document_metadata" list.
        """
        if os.path.exists(self.yaml_file):
            with open(self.yaml_file, 'r') as file:
                return yaml.safe_load(file)
        else:
            return {"document_metadata": []}

    def save_prompts(self):
        """
        Save the current state of the prompts back to the YAML file.
        """
        with open(self.yaml_file, 'w') as file:
            yaml.dump(self.prompts, file)

    def list_prompts(self, category):
        """
        List all available prompts in a given category.

        Args:
            category (str): The category of prompts to list.

        Returns:
            list: A list of all the prompts in the specified category.
        """
        return self.prompts.get(category, [])

    def add_prompt(self, category, new_prompt):
        """
        Add a new prompt to a specific category in the YAML file.

        Args:
            category (str): The category to which the prompt will be added.
            new_prompt (str): The template text of the new prompt to be added.
        """
        if category not in self.prompts:
            self.prompts[category] = []
        self.prompts[category].append({"template": new_prompt})
        self.save_prompts()
        print(f"Prompt added successfully to category '{category}'.")

    def edit_prompt(self, category, index, updated_prompt):
        """
        Edit an existing prompt based on the given index and category.

        Args:
            category (str): The category of the prompt to edit.
            index (int): The index of the prompt to edit.
            updated_prompt (str): The new content to replace the existing prompt.
        """
        try:
            self.prompts[category][index]["template"] = updated_prompt
            self.save_prompts()
            print(f"Prompt in category '{category}' edited successfully.")
        except (IndexError, KeyError):
            print(f"No prompt found at index {index} in category '{category}'.")

    def delete_prompt(self, category, index):
        """
        Delete an existing prompt from a specific category based on the given index.

        Args:
            category (str): The category of the prompt to delete.
            index (int): The index of the prompt to delete.
        """
        try:
            self.prompts[category].pop(index)
            self.save_prompts()
            print(f"Prompt deleted successfully from category '{category}'.")
        except (IndexError, KeyError):
            print(f"No prompt found at index {index} in category '{category}'.")

    def get_prompt(self, category, index):
        """
        Get a specific prompt by its index from a given category.

        Args:
            category (str): The category from which to get the prompt.
            index (int): The index of the prompt to retrieve.

        Returns:
            str: The template of the specified prompt.
            None: If no prompt is found at the given index.
        """
        try:
            return self.prompts[category][index]["template"]
        except (IndexError, KeyError):
            print(f"No prompt found at index {index} in category '{category}'.")
            return None

if __name__ == "__main__":
    prompt_manager = PromptManager()

    # List all prompts in 'document_metadata' category
    # print(prompt_manager.list_prompts('document_metadata')[0]["template"])
    print(prompt_manager.get_prompt("document_metadata",0))
     