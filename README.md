# VisionAssist Online Version

This branch is the online version of our smart glasses project.

It includes features that use online APIs, such as scene description, navigation, weather, and other AI-based functions.

## Setup

Clone the repo:

```bash
git clone https://github.com/mohalsheikh/smart-glasses-project.git
cd smart-glasses-project
git checkout online_version

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
API Keys

Before running, add your own API keys.

You can export them in the terminal like this:

export OPENAI_API_KEY="your_key_here"
export OPENROUTE_API_KEY="your_key_here"
export OPENWEATHER_API_KEY="your_key_here"

Do not use real keys from the repo. Use your own keys.

Run

Run the project with:

python src/controller.py
Notes
This branch is the online version
The main branch is the offline version
Some features will not work unless the needed API keys are added
Camera, microphone, and audio output may also be needed depending on the feature

Then do this:

```bash
git checkout online_version

If your actual branch name is with a dash, use:

git checkout online-version

Then update the README, add .env.example, commit, and push:

cat > .env.example << 'EOF'
OPENAI_API_KEY=your_key_here
OPENROUTE_API_KEY=your_key_here
OPENWEATHER_API_KEY=your_key_here
EOF

git add README.md .env.example
git commit -m "Update online version README and add env example"
git push --set-upstream origin online_version
