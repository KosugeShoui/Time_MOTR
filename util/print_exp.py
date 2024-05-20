
import sys

def main():
    if len(sys.argv) != 2:
        print("実験名を正しく指定してください。")
        return
    
    experiment_name = sys.argv[1]
    print("\nExp Name : ", experiment_name)
    print('\n')

if __name__ == "__main__":
    main()