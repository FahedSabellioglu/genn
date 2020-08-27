def importFasttext():
    try:
        import fasttext as ft
    except ImportError:
        ft = None
    
    if not ft:
        raise ImportError("Couldn't import fasttext.")

    return ft

def importNltk():
    try:
        from nltk.tokenize import word_tokenize as wt
        from nltk.tokenize import punkt
        print("INFO: import nltk.tokenize.word_tokenize function")
        
    except ImportError:
        wt = None
    
    except LookupError:
        raise LookupError("nltk.tokenize.punkt not found.\n"
                          "Please install it using\n"
                          "import nltk\n"
                          "nltk.download('punkt').")

    if not wt:
        raise ImportError("Couldn't import nltk.tokenize.word_tokenize function.")
    
    return wt

