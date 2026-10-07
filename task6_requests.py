import requests

response = requests.get("https://api.github.com/users/torvalds")
data = response.json()
print(data["name"])
print(data["public_repos"])
print(data["bio"])
