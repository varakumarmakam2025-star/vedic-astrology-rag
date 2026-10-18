class ContactBook:
    def __init__(self):
        self.contacts = {}

    def add_contact(self, name, phone, email):
        if name in self.contacts:
            print(f"{name} already exists!")
        else:
            self.contacts[name] = {"phone": phone, "email": email}
            print(f"Added {name}.")

    def search_contact(self, name):
        if name in self.contacts:
            print(f"Phone: {self.contacts[name]['phone']}")
            print(f"Email: {self.contacts[name]['email']}")
        else:
            print(f"{name} not found.")

    def remove_contact(self, name):
        if name in self.contacts:
            del self.contacts[name]
            print(f"Contact '{name}' has been removed.")
        else:
            print(f"{name} is not found")

    def update_contact(self, name, phone=None, email=None):
        if name in self.contacts:
            if phone:
                self.contacts[name]['phone'] = phone
            if email:
                self.contacts[name]['email'] = email
            print(f"Updated {name}.")
        else:
            print(f"{name} not found.")

book = ContactBook()
book.add_contact("sam", "12345", "sam@example.com")
book.add_contact("priya", "67890", "priya@example.com")
book.search_contact("sam")
book.remove_contact("priya")
book.search_contact("priya")
book.update_contact("sam", phone="99999")
book.search_contact("sam")