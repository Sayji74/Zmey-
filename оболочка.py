import база
import sys
import traceback

def run_file(filename):
    """
    Выполняет файл .zmey с обработкой ошибок.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Выполняется файл: {filename}")
        result, error = база.run(filename, text)
        
        if error:
            print(error.as_string())
        elif result:
            if hasattr(result, 'elements') and len(result.elements) == 1:
                print(repr(result.elements[0]))
            else:
                print(repr(result))
    except UnicodeDecodeError:
        print(f"Ошибка: Файл {filename} не в кодировке UTF-8. Пожалуйста, сохраните файл в UTF-8.")
    except FileNotFoundError:
        print(f"Ошибка: Файл {filename} не найден.")
    except Exception as e:
        print(f"Произошла ошибка при выполнении файла {filename}:")
        traceback.print_exc()

def run_interactive():
    """
    Запускает интерактивную оболочку Zмей.
    """
    print("Zмей v1.0 - Интерактивная оболочка. Введите 'выход' для завершения.")
    while True:
        try:
            text = input('Zмей v1.0 > ')
            if text.strip().lower() == "выход":
                print("Завершение работы оболочки.")
                break
            if text.strip() == "":
                continue
            
            result, error = база.run('<stdin>', text)
            
            if error:
                print(error.as_string())
            elif result:
                if hasattr(result, 'elements') and len(result.elements) == 1:
                    print(repr(result.elements[0]))
                else:
                    print(repr(result))
        except KeyboardInterrupt:
            print("\nПрервано пользователем. Введите 'выход' для завершения.")
        except Exception as e:
            print("Произошла ошибка в интерактивной оболочке:")
            traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Если передан аргумент — имя файла .zmey
        run_file(sys.argv[1])
    else:
        # Запуск интерактивной оболочки
        run_interactive()