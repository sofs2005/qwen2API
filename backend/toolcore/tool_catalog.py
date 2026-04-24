from __future__ import annotations

from typing import Iterable

from backend.toolcore.types import ToolDefinition


class ToolCatalog:
    """
    The central source of truth for tool validation and naming.
    
    Request-declared tools are the only allowed tools. The catalog provides:
    - Validation that tool names are declared
    - Name normalization via an alias map (mapping client-facing names to canonical declared names)
    - Reverse mapping from canonical names back to client-facing names (for response generation)
    
    The alias map is scoped to the current request only, so aliases do not persist across requests.
    """

    def __init__(self, tools: Iterable[ToolDefinition], alias_map: dict[str, str] | None = None) -> None:
        """
        Initialize the catalog with declared tools.
        
        Args:
            tools: The tools declared in the current request.
            alias_map: Optional mapping from client-facing names to declared tool names.
                       For example, if client uses "weather-fetcher" but the declared tool is "get_weather",
                       alias_map = {"weather-fetcher": "get_weather"}.
        """
        normalized_tools = []
        for tool in tools:
            client_name = tool.client_name or tool.name
            model_name = tool.model_name or tool.name
            if tool.client_name != client_name or tool.model_name != model_name:
                tool = ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                    client_name=client_name,
                    model_name=model_name,
                    aliases=tool.aliases,
                    raw=tool.raw,
                )
            normalized_tools.append(tool)

        self._tools_by_name: dict[str, ToolDefinition] = {tool.name: tool for tool in normalized_tools}
        self._alias_map: dict[str, str] = alias_map or {}
        for tool in normalized_tools:
            self._alias_map.setdefault(tool.client_name or tool.name, tool.name)
            self._alias_map.setdefault(tool.model_name or tool.name, tool.name)
            for alias in tool.aliases:
                self._alias_map.setdefault(alias, tool.name)

        # Validate that all alias targets exist in declared tools
        for alias, target in self._alias_map.items():
            if target not in self._tools_by_name:
                raise ValueError(f"Alias '{alias}' points to undeclared tool '{target}'")

    def is_declared(self, name: str) -> bool:
        """Return True if a tool with the given name is declared in this request."""
        return name in self._tools_by_name

    def get_canonical_name(self, client_tool_name: str) -> str | None:
        """
        Convert a client-facing tool name to the canonical declared tool name.
        
        If the client name is an alias, return the canonical name.
        If the client name is already a declared tool name, return it as is.
        If the client name is not recognized, return None.
        """
        # Check if it's an alias first
        if client_tool_name in self._alias_map:
            return self._alias_map[client_tool_name]
        # Otherwise, check if it's a declared tool name
        if client_tool_name in self._tools_by_name:
            return client_tool_name
        return None

    def get_client_name(self, canonical_name: str) -> str:
        """
        Convert a canonical tool name (from model response) back to the client-facing name.
        
        If there is a reverse mapping (i.e., some alias points to this canonical name),
        return the *first* alias found. Typically there is only one alias per tool.
        If no alias exists, return the canonical name itself.
        """
        tool_def = self._tools_by_name.get(canonical_name)
        if tool_def is not None:
            return tool_def.client_name or canonical_name
        return canonical_name

    def get_model_name(self, client_tool_name: str) -> str | None:
        canonical = self.get_canonical_name(client_tool_name)
        if canonical is None:
            return None
        tool_def = self._tools_by_name.get(canonical)
        if tool_def is None:
            return canonical
        return tool_def.model_name or canonical

    def validate_tool_choice_name(self, tool_name: str) -> None:
        """
        Validate that a tool name from tool_choice is declared.
        
        Raises ValueError if the tool name is not declared.
        """
        if self.get_canonical_name(tool_name) is None:
            raise ValueError(f"Tool '{tool_name}' undeclared tool")

    def get_tool_definition(self, canonical_name: str) -> ToolDefinition | None:
        """Return the ToolDefinition for the given canonical name, or None if not found."""
        return self._tools_by_name.get(canonical_name)

    def get_all_tool_names(self) -> set[str]:
        """Return the set of all declared tool names (canonical names)."""
        return set(self._tools_by_name.keys())

    def get_all_client_names(self) -> set[str]:
        """
        Return all client-facing names (including aliases and canonical names).
        
        This is useful for early validation of incoming tool names.
        """
        names = set(self._tools_by_name.keys())
        names.update(self._alias_map.keys())
        return names

    def resolve_client_tool_name(self, client_name: str) -> tuple[str, ToolDefinition]:
        """
        Resolve a client-facing tool name to its canonical name and ToolDefinition.
        
        Raises ValueError if the client name is not recognized.
        """
        canonical = self.get_canonical_name(client_name)
        if canonical is None:
            raise ValueError(f"Unknown tool '{client_name}' (not declared and not an alias)")
        tool_def = self._tools_by_name[canonical]
        return canonical, tool_def
