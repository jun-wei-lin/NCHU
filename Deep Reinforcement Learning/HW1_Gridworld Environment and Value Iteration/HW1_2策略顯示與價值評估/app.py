from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    grid = None
    dimension = None
    if request.method == "POST":
        try:
            dimension = int(request.form.get("dimension"))
            if dimension < 5 or dimension > 9:
                raise ValueError("Dimension out of range")
        except:
            return "輸入無效，請輸入介於 5 到 9 的數字。"

        # 建立一個 n×n 的網格，每個格子預設為 "empty"
        grid = [["empty" for _ in range(dimension)] for _ in range(dimension)]
        
    return render_template("index.html", grid=grid, dimension=dimension)

if __name__ == "__main__":
    app.run(debug=True)
