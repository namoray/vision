echo "Some big packages here, sorry for how long this takes... 🥺"
python3.10 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r git_requirements.txt
pip install pre-commit
pre-commit install
echo "Run . venv/bin/activate."
