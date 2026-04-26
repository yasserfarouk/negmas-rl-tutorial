from nbconvert.preprocessors import Preprocessor


class IgnoreCodeCellPreprocessor(Preprocessor):
    def preprocess(self, nb, resources):
        for cell in nb.cells:
            if "tags" in cell.metadata and "hide_code" in cell.metadata["tags"]:
                cell["source"] = ""
        return nb, resources
