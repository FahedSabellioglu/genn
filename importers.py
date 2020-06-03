def importFasttext():
    try:
        import fasttext as ft
        print("Import fasttext")
    except ImportError:
        ft = None
    
    if not ft:
        raise ImportError("Couldn't import fasttext.")

    return ft

def importNltk():
    try:
        from nltk.tokenize import word_tokenize as wt
        print("import nltk.tokenize.word_tokenize function")
    except ImportError:
        wt = None

    if not wt:
        raise ImportError("Couldn't import nltk.tokenize.word_tokenize function.")
    
    return wt

