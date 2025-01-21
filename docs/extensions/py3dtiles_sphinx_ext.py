from sphinx.application import Sphinx


def generate_js_file(app: Sphinx, pagename: str, templatename: str, context, doctree):
    # Access the custom setting from `app.config`
    current_version = app.config.smv_current_version

    # JavaScript template
    js = f"""
    // JavaScript generated from template
    DOCUMENTATION_OPTIONS.theme_switcher_version_match = "{current_version}"
    """
    app.add_js_file(None, body=js)

    # # Output path for the JavaScript file
    # output_path = os.path.join(app.outdir, '_static', 'py3dtiles.js')
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # with open(output_path, 'w') as js_file:
    #     js_file.write(js_template)


def setup(app):
    """Setup the Sphinx extension."""
    # Connect the JavaScript generation to the build process
    app.connect("html-page-context", generate_js_file, priority=1000)
    return {"version": "0.1", "parallel_read_safe": True}
