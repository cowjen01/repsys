from repsys import Dataset


class MovieLens(Dataset):
    def name(self):
        return "movielens"

    def get_genres(self):
        return (
            self.items["subtitle"]
            .dropna()
            .str.split(", ", expand=True)
            .stack()
            .unique()
        )
