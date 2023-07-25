# app.py (Backend)

from flask import Flask, render_template, request, send_file, redirect, url_for

import os

from .mlflow_integration import attempt_upload, get_files_with_info
from .mlflow_integration import delete_mlflow_file, get_info
from .mlflow_integration import get_history, get_all_trees_with_links

app = Flask(__name__)

STORAGE_PATH = os.environ.get("MODEL_STORAGE_PATH")


# Endpoint to handle file uploads
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        issues = attempt_upload(request.files['file'],
                                request.form.get("model_name"),
                                request.form.get("overwrite") == "1",
                                STORAGE_PATH)
        if issues:
            return issues
        return "File uploaded successfully!"
    return render_template("upload.html")


# Endpoint to browse and download files along with custom information
@app.route("/files")
def browse_files():
    files_with_info = get_files_with_info()
    return render_template("browse_files.html", files=files_with_info)


@app.route("/delete/<filename>")
def delete_file(filename):
    delete_mlflow_file(filename, STORAGE_PATH)
    return redirect(url_for("browse_files"))


# Endpoint for info for specific file
@app.route("/info/<filename>")
def show_file_info(filename):
    file_path = os.path.join(STORAGE_PATH, filename)
    return render_template("file_info.html", info=get_info(file_path))


# Endpoint for history of a specific file
@app.route("/history/<filename>")
def show_file_history(filename):
    file_path = os.path.join(STORAGE_PATH, filename)
    return render_template("history.html", history=get_history(file_path))


@app.route("/download/<filename>")
def download_file(filename):
    return send_file(filename, as_attachment=True)


@app.route("/")
def landing_page():
    return render_template("landing.html")


@app.route("/all_trees")
def all_trees():
    all_trees_with_links = get_all_trees_with_links(STORAGE_PATH)
    return render_template("all_trees.html",
                           all_trees_with_links=all_trees_with_links)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
