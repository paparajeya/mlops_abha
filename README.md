This is a workshop MLOps repository.

## How to configure and run the project
1. **Fork the repository**: Create your own copy of the repository on GitHub.
2. **Clone the repository**: Use `git clone <your-fork-url>` to clone the repository to your local machine.
3. **Install dependencies**: Navigate to the project directory and run `pip install -r requirements.txt` to install the necessary Python packages.
4. **Run the training script**: Execute `python app/train.py` to train the model.
5. **Build the Docker image**: Run `docker build -t mlops_abha .` to build the Docker image.
6. **Run the Docker container**: Use `docker run -p 5050:5050 mlops_abha` to start the container and expose the API on port 5050.
7. **Access the API**: Open your web browser and navigate to `http://localhost:5050` to access the Flask API.


### ✅ Conventional Commit Message Structure

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

* **`type`**: Indicates the nature of the change. Common types include:

  * `feat`: A new feature
  * `fix`: A bug fix
  * `docs`: Documentation changes
  * `style`: Code style changes (formatting, missing semi-colons, etc.)
  * `refactor`: Code changes that neither fix a bug nor add a feature
  * `perf`: Performance improvements
  * `test`: Adding or correcting tests
  * `chore`: Maintenance tasks (build process, auxiliary tools, libraries)

* **`scope`** *(optional)*: Provides additional contextual information, typically the name of the affected module or component, enclosed in parentheses.

* **`description`**: A concise summary of the change, written in the imperative mood (e.g., "add", "fix", "update").

* **`body`** *(optional)*: A more detailed explanation of the change, its rationale, and any relevant background information.

* **`footer(s)`** *(optional)*: Additional metadata, such as issue references or notes about breaking changes.

### 🛠 Examples

#### 1. **Adding a New Feature**

```
feat(auth): add OAuth2 login support
```

*Adds a new feature to the authentication module.*

#### 2. **Fixing a Bug**

```
fix(api): resolve null pointer exception on user creation
```

*Fixes a bug in the API module related to user creation.*

#### 3. **Documentation Update**

```
docs(readme): update installation instructions
```

*Updates the installation instructions in the README.*

#### 4. **Code Refactoring**

```
refactor(database): optimize query performance
```

*Refactors database queries for improved performance.*

#### 5. **Performance Improvement**

```
perf(image-processing): reduce image loading time
```

*Improves the performance of image loading in the image-processing module.*

#### 6. **Test Addition**

```
test(auth): add unit tests for login functionality
```

*Adds unit tests for the login functionality in the authentication module.*

#### 7. **Chore Task**

```
chore(deps): update dependency versions
```

*Updates project dependencies to their latest versions.*

#### 8. **Breaking Change**

If a commit introduces a breaking change, indicate it by adding an exclamation mark (`!`) after the type or scope, and include a `BREAKING CHANGE` footer:

```
feat(auth)!: remove legacy authentication methods

BREAKING CHANGE: The legacy authentication methods have been removed. Users must now use OAuth2.
```

*Introduces a breaking change by removing legacy authentication methods.*

### 🔗 Linking to Issues

To associate commits with issues, include references in the footer:

```
fix(auth): correct password reset link

Resolves: #123
```

*Fixes issue number 123 related to password reset links.*

### 📌 Tips for Effective Commit Messages

* **Use the imperative mood**: Write as if you're giving a command (e.g., "add feature", not "added feature").
* **Be concise**: Keep the subject line under 50 characters if possible.
* **Provide context**: Use the body to explain the "why" behind the change.
* **Consistent formatting**: Maintain a consistent structure to facilitate automated tools.
* **Review before committing**: Always review your commit message for clarity and completeness.