import functools
import re

from nltk.corpus import wordnet


def replace_with_nothing(text, replacements):
    for replacement in replacements:
        text = text.replace(replacement, "")
    return text


def translate_tag(tag):
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    new_tag = tag_dict.get(tag[0])
    if new_tag is None:
        new_tag = wordnet.NOUN
    return new_tag


def compose(*functions):
    """
    Composes an arbitrary number of functions.\n
    Given two functions f and g for example, compose returns a function
        h = g ∘ f
    such that
        h(args) = g(f(args)

    :return: fn ∘ ... ∘ f2 ∘ f1
    """
    if len(functions) == 0:
        return lambda x: x
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def remove_quotes(text):
    """
    Removes quotes from the given input text and returns the result.

    :param text: The input text that will be preprocessed
    :return: The input text without quotation-marks
    """
    return replace_with_nothing(text, ["`", '"', "¨", "'", "`", "´"])


def lower(text):
    """
    Converts all uppercase characters from text into lowercase characters and returns it.

    :param text: The input text that will be preprocessed
    :return: The lowercase input string
    """
    return text.lower()


def remove_punctuation(text):
    """
    Removes every punctuation-marks from text and returns the result.
    :param text: The input text that will be preprocessed
    :return: The input text without punctuation-marks
    """
    return replace_with_nothing(text, [".", ",", "!", "?", ":", ";"])


def remove_short_tokens(text):

    return re.sub(r" (.|..) ", " ", text)
