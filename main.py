from agents import run_user_query


def main() -> None:
    print("Career Assistant CLI started. Type 'exit' to quit.")
    while True:
        query = input("\n请输入你的问题：").strip()
        if query.lower() == "exit":
            print("已退出。")
            break
        if not query:
            print("输入不能为空，请重新输入。")
            continue

        try:
            result = run_user_query(query)
            print("\n分类：", result.get("category", ""))
            print("输出文件：", result.get("response", ""))
        except Exception as exc:
            print(f"\n运行出错：{exc}")
            print("请检查 .env 的 DASHSCOPE_API_KEY 与依赖安装后重试。")


if __name__ == "__main__":
    main()
