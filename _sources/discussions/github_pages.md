# Publishing to GitHub Pages

## What is GitHub Pages?

[GitHub Pages](https://pages.github.com/) is a free static site hosting service provided by GitHub. It serves HTML, CSS, and JavaScript files directly from a repository, making it an easy way to publish documentation, portfolios, or project pages to the web.

Key features:
- **Free hosting** for static content
- **Custom domains** supported (or use `<username>.github.io/<repo-name>`)
- **Works with private repos** — you can keep your code private while making the Pages site public
- **No server-side code** — just static files (HTML, CSS, JS, images)

In this course, we use GitHub Pages to publish analysis results and project outputs, giving you practice with a common workflow for sharing research deliverables.


## Publishing a Static Site

Follow these steps to publish any HTML file (an exported Jupyter notebook, a Sphinx build, a hand-written page, etc.) to GitHub Pages.

### Step 1: Create or Export Your HTML

Generate the HTML you want to publish. A few common sources:

- **Jupyter notebook**: In VS Code, click the "..." menu in the top-right corner of the notebook, then **Export > HTML**. In JupyterLab, use **File > Save and Export Notebook As > HTML**.
- **Sphinx / MkDocs / Quarto**: Run the build command and grab the output directory.
- **Plain HTML**: Write it by hand or with a template.

Name the file you want to be the landing page `index.html`.

### Step 2: Set Up the `./docs/` Folder

In your repository:

1. Create a `docs/` folder in the repo root
2. Place your `index.html` file inside `docs/`
3. Create an empty file called `.nojekyll` in `docs/`

The `.nojekyll` file tells GitHub not to process your files with Jekyll (GitHub's default static site generator). This is important because Jekyll can interfere with certain file names and paths (e.g., files or folders that start with an underscore).

Your folder structure should look like:

```
your-repo/
├── docs/
│   ├── index.html
│   └── .nojekyll
├── src/
│   └── ...
└── ...
```

### Step 3: Commit and Push

```bash
git add docs/
git commit -m "Publish site to GitHub Pages"
git push
```

You can also do this using GitKraken, VS Code's Source Control panel, or GitHub Desktop.

### Step 4: Enable GitHub Pages

1. Go to your repository on GitHub
2. Navigate to **Settings > Pages**
3. Under "Source", select **Deploy from a branch**
4. Set Branch to `main` and Folder to `/docs`
5. Click **Save**

After a few minutes, your site will be live at:

```
https://<org-or-username>.github.io/<repo-name>/
```

**Note**: Your repository can remain private. Only the GitHub Pages site will be publicly accessible.


## Tips and Gotchas

- **First build can take a couple of minutes.** If your site 404s right after enabling Pages, wait and refresh.
- **Cache is aggressive.** When you push updates, do a hard refresh (Cmd+Shift+R) to see changes.
- **Use relative paths** for any images, CSS, or JS in your HTML so the site works under the `/<repo-name>/` subpath.
- **Check the Actions tab** on GitHub — Pages builds show up there, and failures will tell you what went wrong.
