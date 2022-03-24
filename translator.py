import sys

import deepl


def __read_auth_key(path):
    file = open(path, "r")
    return "".join(file.readlines())  # return full file as a single string


translator = deepl.Translator(__read_auth_key("deepl.auth"))


def print_info():
    print("DeepL Info:")
    # Check account usage
    usage = translator.get_usage()
    print(f"  Character usage: {usage.character.count} of {usage.character.limit} = {round((usage.character.count*100)/usage.character.limit, 1)}%\n")

    # Source and target languages
    print("  Source languages:")
    for language in translator.get_source_languages():
        print(f"    {language.code} ({language.name})")  # Example: "DE (German)"
    print("")

    print("  Target languages:")
    for language in translator.get_target_languages():
        if language.supports_formality:
            print(f"    {language.code} ({language.name}) supports formality")
        else:
            print(f"    {language.code} ({language.name})")
    print("")


def translate(source_text, target_lang="EN-US"):
    result = translator.translate_text(source_text, target_lang=target_lang)
    return result


if __name__ == "__main__":
    print_info()
    sys.exit(0)
