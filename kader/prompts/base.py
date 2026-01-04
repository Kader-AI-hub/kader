import os
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


class PromptBase:
    """
    Base class for handling Jinja2 templates for prompts.

    Attributes
    ----------
    template : str, optional
        A string containing the Jinja2 template.
    template_path : str, optional
        The path to the Jinja2 template file.
    """

    template: str | None = None
    template_path: str | None = None

    def __init__(self, **kwargs) -> None:
        """Initialize the PromptBase instance.

        Parameters
        ----------
        **kwargs
            Keyword arguments that will be available as variables in the template.
            Special keys 'template' and 'template_path' are used to initialize the prompt.
        """
        self.template = kwargs.pop("template", self.template)
        self.template_path = kwargs.pop("template_path", self.template_path)
        self.vars = kwargs

        if self.template:
            env = Environment()
            self.prompt = env.from_string(self.template)
        elif self.template_path:
            current_dir = Path(__file__).parent
            path = os.path.join(current_dir, "templates")
            loader = FileSystemLoader(path)
            env = Environment(loader=loader)
            self.prompt = env.get_template(self.template_path)
        self._resolved_prompt = None

    def render_template(self) -> str:
        """Render the Jinja2 template with the provided variables.

        Returns
        -------
        str
            The rendered prompt as a string.
        """
        render = self.prompt.render(**self.vars)
        render = re.sub(r"\n{3,}", "\n\n", render)
        return render

    def resolve_prompt(self) -> str:
        """Resolve the prompt by rendering it if it hasn't been already.

        This method ensures the template is rendered only once and the result is cached.

        Returns
        -------
        str
            The resolved prompt.
        """
        if self._resolved_prompt is None:
            self._resolved_prompt = self.render_template()
        return self._resolved_prompt

    def __str__(self) -> str:
        """Return the resolved prompt when the object is converted to a string.

        Returns
        -------
        str
            The resolved prompt.
        """
        return self.resolve_prompt()
