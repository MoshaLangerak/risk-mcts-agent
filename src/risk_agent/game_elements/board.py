import json


class Board:
    def __init__(self) -> None:
        """
        Initialise the board with default values.
        """
        self.territories: dict[int, dict] = {}
        self.adjacency_list: dict[int, list[int]] = {}
        self.continents: dict[int, dict] = {}

    def load_from_file(self, file_path: str) -> None:
        """
        Loads the board configuration from a JSON file.
        """
        with open(file_path) as f:
            data = json.load(f)

        for continent in data.get('continents', []):
            self.continents[continent['id']] = {
                'name': continent['name'],
                'bonus': continent['bonus'],
                'territories': continent.get('territory_ids', []),
            }

        for territory in data.get('territories', []):
            self.territories[territory['id']] = {
                'name': territory['name'],
                'continent': territory['continent_id'],
            }

            self.adjacency_list[territory['id']] = territory.get('adjacencies', [])

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Board):
            return False
        return (
            self.territories == value.territories
            and self.adjacency_list == value.adjacency_list
            and self.continents == value.continents
        )

    def copy(self) -> 'Board':
        """
        Create a deep copy of the board.
        """
        new_board = Board()
        new_board.territories = self.territories.copy()
        new_board.adjacency_list = self.adjacency_list.copy()
        new_board.continents = self.continents.copy()
        return new_board
