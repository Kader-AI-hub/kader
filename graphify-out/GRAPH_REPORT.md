# Graph Report - kader  (2026-10-02)

## Corpus Check
- 182 files · ~166,273 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 20 file(s) not represented in the graph (top: .j2 7, (none) 6, .mmd 6)

## Summary
- 3852 nodes · 7842 edges · 204 communities (67 shown, 137 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 572 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- LLM Provider Base
- Agent Examples
- Memory & Conversation
- OpenAI Compatible Provider
- Anthropic Provider Examples
- Mistral Provider Examples
- Google Provider Examples
- Ollama Provider Examples
- Provider Demo Patterns
- Filesystem Tools Tests
- CLI Callbacks & Sessions
- Tool Call Handling
- Configuration & Google GenAI
- Command Execution Tool
- Anthropic Provider Impl
- Base Agent Core
- CLI Subagent Tracker
- Filesystem Backend Tools
- Agent State Management
- Session Persistence
- Tool Schema & Parameters
- Async Memory Tests
- Tool Base Patterns
- CLI App Main
- Session Save/Load
- Documentation Concepts
- CLI Settings
- Test Infrastructure
- Async Session Manager
- Web Tools Tests
- OpenAI Compatible Impl
- Tool Base Patterns
- Todo Metadata Tests
- CLI Settings Init
- Backend Protocol
- Todo Agent Example
- Agent Tool
- Memory Compression
- Filesystem Backend
- CLI Commands Base
- Memory Example
- Kader Init & Checkpointer
- Subagent Loader
- Base Tool Execution
- Planner Executor Workflow
- File Session Manager
- Dev Helper Scripts
- Todo Metadata
- Dev Helper Scripts
- Agent Logger Tests
- Skills Tool
- Context Aggregator
- LLM Provider Factory
- CLI Screenshots
- Kader App CLI
- Hello Skill Example
- Agent Logger
- CLI Utils
- CLI Screenshots
- Agent Prompts
- OpenAI Compatible Tests
- Filesystem Hardening Tests
- GitHub Workflows
- Hierarchical Conversation
- Todo Tool Tests
- Agent Tool Skills Tests
- Tool Execution
- Gitignore Utils
- File Memory Tests
- Tool Loader
- Command Callback Example
- Filesystem Tools Impl
- Agent Tool Skills Tests
- Tool Protocol
- CLI Session Metadata
- Skills Tool Patterns
- Graphify References
- Update Command
- Conversation Manager
- Agent Tool Config
- Base Workflow
- Agent Logger Integration
- Agent Tool Tests
- CLI Tools Init
- Kader Config
- Provider Base Tests
- File Memory Tests
- Filesystem Hardening Tests
- CLI Design HTML
- CLI Screenshots
- Initialize Command
- Documentation Agents
- Persistent Conversation
- Filesystem Tools Tests
- Settings Migration
- Memory Config
- Subagent Workflows
- Anthropic Provider Tests
- Filesystem Tools Tests
- Filesystem Hardening Tests
- Skills Tests
- Tools Base Tests
- Graphify Transcribe
- Connect Command
- Callbacks Documentation
- Memory Documentation
- Python Developer Example
- Filesystem Tools Tests
- Filesystem Tools Tests
- Contributing Guide
- CLI App Sessions
- Agent Tool Execution
- Todo Metadata Tests
- Filesystem Tools Tests
- Filesystem Tools Tests
- Filesystem Tools Tests
- Planner Prompt
- Agent Tool Tests
- Graphify Query
- CLI Screenshots
- CLI README
- CLI Image Docs
- README Docs
- Configuration Docs
- Providers Documentation
- Base Agent YAML
- Backend Protocol Upload
- Backend Protocol Write
- Mistral Provider Tests
- OpenAI Compatible Tests
- Provider Base Tests
- File Memory Tests
- Agent Tool Tests
- Filesystem Tools Tests
- Filesystem Tools Tests
- Skills Tests
- Todo Tool
- Documentation Agents
- Calculator Skill
- Hello Skill
- Provider README
- CLI App Callback
- Simple Agent Example
- Todo Status
- File Memory Tests
- Todo Tool Tests
- Banner Type
- Memory README
- Callbacks README
- Opencode Plugin
- Filesystem Tools Tests
- CLI Screenshots
- Tool Call & Result
- CLI Screenshots
- Mistral Provider Docs
- Command Loader Docs
- Tool Output Compressor
- Read Directory Tool
- Replace Lines Tool
- Web Fetch Tool
- Web Search Tool
- Package Kader

## God Nodes (most connected - your core abstractions)
1. `Message` - 178 edges
2. `ModelConfig` - 152 edges
3. `OpenAICompatibleProvider` - 102 edges
4. `Usage` - 95 edges
5. `OpenAIProviderConfig` - 85 edges
6. `AnthropicProvider` - 71 edges
7. `AgentTool` - 71 edges
8. `BaseTool` - 71 edges
9. `BaseAgent` - 68 edges
10. `BaseLLMProvider` - 67 edges

## Surprising Connections (you probably didn't know these)
- `Contributing to Kader Skill (.kader)` --semantically_similar_to--> `Contributing to Kader Skill (.opencode)`  [INFERRED] [semantically similar]
  .kader/skills/contributing-to-kader/SKILL.md → .opencode/skills/contributing-to-kader/SKILL.md
- `Contributor Checklist (.kader)` --semantically_similar_to--> `Contributor Checklist (.opencode)`  [INFERRED] [semantically similar]
  .kader/skills/contributing-to-kader/assets/contributor_checklist.md → .opencode/skills/contributing-to-kader/assets/contributor_checklist.md
- `Kader Agent Instructions (.kader)` --semantically_similar_to--> `Kader Agent Instructions (.opencode)`  [INFERRED] [semantically similar]
  .kader/skills/contributing-to-kader/references/kader_agent_instructions.md → .opencode/skills/contributing-to-kader/references/kader_agent_instructions.md
- `Workflows` --semantically_similar_to--> `Planner-Executor Framework`  [INFERRED] [semantically similar]
  docs/core-framework/index.md → README.md
- `Planner-Executor Framework` --semantically_similar_to--> `PlannerExecutorWorkflow`  [INFERRED] [semantically similar]
  README.md → cli/README.md

## Import Cycles
- 3-file cycle: `cli/app.py -> cli/commands/__init__.py -> cli/commands/base.py -> cli/app.py`
- 4-file cycle: `cli/app.py -> cli/commands/__init__.py -> cli/commands/refresh.py -> cli/commands/base.py -> cli/app.py`
- 4-file cycle: `cli/app.py -> cli/commands/__init__.py -> cli/commands/update.py -> cli/commands/base.py -> cli/app.py`
- 4-file cycle: `cli/app.py -> cli/commands/__init__.py -> cli/commands/connect.py -> cli/commands/base.py -> cli/app.py`
- 4-file cycle: `cli/app.py -> cli/commands/__init__.py -> cli/commands/initialize.py -> cli/commands/base.py -> cli/app.py`

## Hyperedges (group relationships)
- **CI/CD Pipeline** — _github_workflows_ci_workflow, _github_workflows_pages_workflow, _github_workflows_release_workflow [INFERRED 0.85]
- **Contributing to Kader Skill Package** — _kader_skills_contributing_to_kader_skill_contributing_to_kader_skill, _kader_skills_contributing_to_kader_assets_contributor_checklist_contributor_checklist, _kader_skills_contributing_to_kader_references_kader_agent_instructions_kader_agent_instructions [INFERRED 0.85]
- **Graphify Reference Documentation** — _opencode_skills_graphify_references_add_watch_add_watch_reference, _opencode_skills_graphify_references_exports_exports_reference, _opencode_skills_graphify_references_extraction_spec_extraction_spec_reference, _opencode_skills_graphify_references_github_and_merge_github_and_merge_reference, _opencode_skills_graphify_references_hooks_hooks_reference [INFERRED 0.85]
- **Core Framework Pillars** — docs_core_framework_index_providers, docs_core_framework_index_tools, docs_core_framework_index_agents, docs_core_framework_index_memory, docs_core_framework_index_workflows [INFERRED 0.85]
- **Kader CLI Input Modes** — assets_design_v2_code_command_mode, assets_design_v2_code1_prompt_mode, assets_design_v2_code3_system_mode [INFERRED 0.85]
- **Callback Class Hierarchy** — docs_core_framework_callbacks_basecallback, docs_core_framework_callbacks_toolcallback, docs_core_framework_callbacks_llmcallback, docs_core_framework_callbacks_loggingtoolcallback, docs_core_framework_callbacks_loggingllmcallback [EXTRACTED 1.00]
- **Skills System** — examples_skills_skills_calculator_skill_calculator, examples_skills_skills_github_skill_github, examples_skills_skills_hello_skill_hello, examples_skills_skills_joke_skill_joke, examples_skills_skills_visualization_skill_visualization, docs_core_framework_tools_skillstool [INFERRED 0.85]
- **LLM Provider Interface** — kader_readme_ollamaprovider, kader_readme_googleprovider, kader_readme_anthropicprovider, kader_readme_mistralprovider, kader_readme_openaicompatibleprovider [INFERRED 0.85]
- **Planner-Executor Architecture** — docs_core_framework_subagents_plannerexecutorworkflow, docs_core_framework_subagents_subagentloader, docs_core_framework_subagents_agenttool, docs_core_framework_subagents_reactagent, docs_core_framework_subagents_contextaggregator, kader_readme_planningagent [INFERRED 0.85]
- **Kader CLI Three-Panel Layout** — assets_design_v2_screen_files_panel, assets_design_v2_screen_plan_panel, assets_design_v2_screen_chat_area [EXTRACTED 1.00]
- **Command Input to Palette Suggestions Flow** — assets_design_v2_screen_command_input, assets_design_v2_screen_command_palette, assets_design_v2_screen_chat_area [EXTRACTED 1.00]
- **Kader CLI Three-Panel Layout** — assets_design_v2_screen1_files_panel, assets_design_v2_screen1_plan_panel, assets_design_v2_screen1_chat_interface [EXTRACTED 1.00]
- **Chat-to-Provider Message Flow** — assets_design_v2_screen1_chat_interface, assets_design_v2_screen1_prompt_mode, assets_design_v2_screen1_ollama_provider, assets_design_v2_screen1_error_handling [EXTRACTED 1.00]
- **Kader CLI slash command set** — assets_design_v2_screen2_command_help, assets_design_v2_screen2_command_clear, assets_design_v2_screen2_command_load, assets_design_v2_screen2_command_cost, assets_design_v2_screen2_command_models, assets_design_v2_screen2_command_save, assets_design_v2_screen2_command_sessions, assets_design_v2_screen2_command_exit [EXTRACTED 1.00]
- **Ollama connection failure flow** — assets_design_v2_screen2_user_message, assets_design_v2_screen2_thinking_indicator, assets_design_v2_screen2_ollama_error, assets_design_v2_screen2_ollama_provider [INFERRED 0.85]
- **Kader CLI Main Layout** — assets_design_v2_screen3_files_panel, assets_design_v2_screen3_plan_panel, assets_design_v2_screen3_chat_interface, assets_design_v2_screen3_command_palette [EXTRACTED 1.00]

## Communities (204 total, 137 thin omitted)

### Community 0 - "LLM Provider Base"
Cohesion: 0.02
Nodes (14): CostInfo, LLMResponse, MessageRole, StreamChunk, Usage, MockLLM, TestMockLLM, test_async() (+6 more)

### Community 1 - "Agent Examples"
Cohesion: 0.03
Nodes (30): main(), main(), demo_agent_tool(), demo_async_operations(), async_demo(), demo_custom_tool(), demo_file_system_tools(), demo_tool_parameters() (+22 more)

### Community 2 - "Memory & Conversation"
Cohesion: 0.02
Nodes (10): main(), CompressionConfig, ConversationMessage, NullConversationManager, SlidingWindowConversationManager, ConversationSummary, HierarchicalConversationManager, PersistentSlidingWindowConversationManager (+2 more)

### Community 3 - "OpenAI Compatible Provider"
Cohesion: 0.03
Nodes (13): demo_async_invocation(), async_demo(), OpenAICompatibleProvider, OpenAIProviderConfig, TestDeepSeekV4ThinkingMode, TestOpenAICompatibleProviderConvertConfig, TestOpenAICompatibleProviderConvertMessages, TestOpenAICompatibleProviderCountTokens (+5 more)

### Community 4 - "Anthropic Provider Examples"
Cohesion: 0.04
Nodes (18): demo_async_invocation(), demo_async_streaming(), async_stream_demo(), demo_basic_invocation(), demo_configuration(), demo_conversation_history(), demo_cost_estimation(), demo_list_models() (+10 more)

### Community 5 - "Mistral Provider Examples"
Cohesion: 0.03
Nodes (17): demo_async_invocation(), demo_async_streaming(), async_stream_demo(), demo_basic_invocation(), demo_configuration(), demo_cost_estimation(), demo_list_models(), demo_model_info() (+9 more)

### Community 6 - "Google Provider Examples"
Cohesion: 0.04
Nodes (15): demo_async_invocation(), async_demo(), demo_async_streaming(), async_stream_demo(), demo_basic_invocation(), demo_configuration(), demo_conversation_history(), demo_cost_estimation() (+7 more)

### Community 7 - "Ollama Provider Examples"
Cohesion: 0.04
Nodes (10): demo_async_invocation(), demo_async_streaming(), async_stream_demo(), demo_basic_invocation(), demo_configuration(), demo_error_handling(), demo_streaming(), main() (+2 more)

### Community 8 - "Provider Demo Patterns"
Cohesion: 0.05
Nodes (21): async_demo(), demo_tool_calling(), demo_conversation_manager(), async_demo(), demo_conversation_history(), async_demo(), demo_conversation_history(), demo_configuration() (+13 more)

### Community 9 - "Filesystem Tools Tests"
Cohesion: 0.03
Nodes (13): temp_dir(), TestEditFileTool, run_test(), TestGlobTool, run_test(), TestGrepTool, run_test(), TestReadDirectoryTool (+5 more)

### Community 10 - "CLI Callbacks & Sessions"
Cohesion: 0.05
Nodes (8): load_callbacks_from_settings(), BaseCallback, CallbackEvent, LLMCallback, LoggingLLMCallback, CallbackLoader, LoggingToolCallback, ToolCallback

### Community 11 - "Tool Call Handling"
Cohesion: 0.04
Nodes (5): ToolCall, ConcreteTool, TestBaseTool, __init__(), TestToolCall

### Community 12 - "Configuration & Google GenAI"
Cohesion: 0.05
Nodes (7): main(), BaseLLMProvider, ModelInfo, ModelPricing, _is_retryable_error(), agenerate_session_title(), generate_session_title()

### Community 13 - "Command Execution Tool"
Cohesion: 0.05
Nodes (3): demo_command_execution(), CommandExecutorTool, TestCommandExecutorTool

### Community 14 - "Anthropic Provider Impl"
Cohesion: 0.05
Nodes (10): ModelConfig, TestAnthropicProviderConvertConfig, TestModelConfig, EchoTool, main(), MockLLMProvider, test_agent_mock_invocation(), test_agent_structure_and_yaml() (+2 more)

### Community 16 - "CLI Subagent Tracker"
Cohesion: 0.05
Nodes (5): SubagentTrackerCallback, main(), MyAgentCallback, MyLLMCallback, CallbackContext

### Community 17 - "Filesystem Backend Tools"
Cohesion: 0.07
Nodes (17): GrepMatch, build_grep_results_dict(), check_empty_content(), create_file_data(), file_data_to_string(), format_content_with_line_numbers(), format_grep_matches(), _format_grep_results() (+9 more)

### Community 23 - "CLI App Main"
Cohesion: 0.07
Nodes (5): enter_fullscreen(), exit_fullscreen(), format_plan_display(), main(), main()

### Community 24 - "Session Save/Load"
Cohesion: 0.10
Nodes (8): aload_json(), asave_json(), awrite_text(), decode_bytes_values(), encode_bytes_values(), get_timestamp(), load_json(), save_json()

### Community 25 - "Documentation Concepts"
Cohesion: 0.06
Nodes (36): ContextAggregator, PlannerExecutorWorkflow, Subagent, SubagentConfig, SubagentLoader, BaseTool, CommandExecutorTool, GlobTool (+28 more)

### Community 26 - "CLI Settings"
Cohesion: 0.09
Nodes (5): KaderSettings, TestKaderSettingsDefaults, TestKaderSettingsSerialisation, TestKaderSettingsValidation, TestModelStringHelpers

### Community 30 - "OpenAI Compatible Impl"
Cohesion: 0.08
Nodes (4): _detect_provider(), _generate_session_id(), TestProviderDetection, TestGenerateSessionId

### Community 32 - "Todo Metadata Tests"
Cohesion: 0.09
Nodes (15): test_compute_todo_stats_all_completed(), test_compute_todo_stats_empty_list(), test_compute_todo_stats_none_completed(), test_compute_todo_stats_partial_completed(), test_handle_create(), test_handle_create_with_empty_items(), test_handle_delete(), test_handle_update() (+7 more)

### Community 33 - "CLI Settings Init"
Cohesion: 0.13
Nodes (9): ensure_settings_file(), get_settings_path(), load_settings(), migrate_settings(), _migrate_user_callbacks(), _migrate_user_tools(), save_settings(), model_cmd() (+1 more)

### Community 34 - "Backend Protocol"
Cohesion: 0.07
Nodes (3): BackendProtocol, EditResult, FileDownloadResponse

### Community 36 - "Agent Tool"
Cohesion: 0.10
Nodes (4): AgentTool, TestAgentToolPersistence, TestAgentToolSkillsInit, TestAgentToolInit

### Community 37 - "Memory Compression"
Cohesion: 0.09
Nodes (3): compress_tool_output(), CompressionStrategy, ToolOutputCompressor

### Community 39 - "CLI Commands Base"
Cohesion: 0.11
Nodes (4): BaseCommand, ConnectCommand, RefreshCommand, UpdateCommand

### Community 40 - "Memory Example"
Cohesion: 0.08
Nodes (5): demo_agent_state(), demo_full_workflow(), demo_request_state(), demo_session_manager(), RequestState

### Community 43 - "Subagent Loader"
Cohesion: 0.18
Nodes (5): load_subagents_from_settings(), SubagentLoader, create_subagent_yaml(), TestLoadSubagentsFromSettings, TestSubagentLoader

### Community 46 - "File Session Manager"
Cohesion: 0.11
Nodes (3): FileSessionManager, Session, SessionType

### Community 47 - "Dev Helper Scripts"
Cohesion: 0.16
Nodes (13): all_checks(), dev_mode(), format_check(), format_code(), lint_code(), lint_fix(), main(), run_cli() (+5 more)

### Community 49 - "Dev Helper Scripts"
Cohesion: 0.16
Nodes (13): all_checks(), dev_mode(), format_check(), format_code(), lint_code(), lint_fix(), main(), run_cli() (+5 more)

### Community 55 - "CLI Screenshots"
Cohesion: 0.13
Nodes (21): ATIS Dataset Files, Chat Conversation Area, /clear Command, Command Input with Slash Prefix, Command Palette Suggestions, Conversation Cleared Notification, /cost Command, diet_data Folder (+13 more)

### Community 57 - "Hello Skill Example"
Cohesion: 0.13
Nodes (3): main(), SkillLoader, TestSkillLoader

### Community 59 - "CLI Utils"
Cohesion: 0.13
Nodes (4): CLICommand, get_commands_text(), get_special_commands(), CommandLoader

### Community 60 - "CLI Screenshots"
Cohesion: 0.13
Nodes (19): /clear command, /cost command, /exit command, /help command, /load command, /models command, /save command, /sessions command (+11 more)

### Community 63 - "Agent Prompts"
Cohesion: 0.20
Nodes (7): BasicAssistancePrompt, CommandAgentPrompt, ExecutorAgentPrompt, PlanningAgentPrompt, ReActAgentPrompt, SessionTitlePrompt, PromptBase

### Community 65 - "Filesystem Hardening Tests"
Cohesion: 0.12
Nodes (3): TestAtomicWrites, capturing_replace(), boom()

### Community 66 - "GitHub Workflows"
Cohesion: 0.17
Nodes (17): GitHub Actions, CI Workflow, opencode Workflow, MkDocs Documentation Builder, Deploy Documentation Workflow, Release Workflow, Lint and Test Command, KADER.md Agent Instructions (+9 more)

### Community 68 - "Todo Tool Tests"
Cohesion: 0.12
Nodes (8): test_create_todo(), test_delete_todo(), test_read_todo_not_found(), test_session_id_override(), test_update_todo_integrity_add_task(), test_update_todo_integrity_modify_task(), test_update_todo_integrity_remove_task(), test_update_todo_status_only()

### Community 72 - "Tool Execution"
Cohesion: 0.16
Nodes (4): ToolExecutionRejected, TestAgentToolRejection, async_invoke_rejected(), TestToolExecutionRejected

### Community 73 - "Gitignore Utils"
Cohesion: 0.22
Nodes (5): filter_by_gitignore(), get_gitignore_filter(), is_ignored(), _match_gitignore_pattern(), _parse_gitignore_patterns()

### Community 76 - "Command Callback Example"
Cohesion: 0.14
Nodes (3): GitRtkCallback, main(), _get_pty_module()

### Community 78 - "Agent Tool Skills Tests"
Cohesion: 0.14
Nodes (3): _create_skill(), TestAgentToolSkillsAsyncExecution, TestAgentToolSkillsExecution

### Community 83 - "Graphify References"
Cohesion: 0.15
Nodes (13): Add-Watch Reference, Exports Reference, Extraction-Spec Reference, GitHub-and-Merge Reference, Hooks Reference, AST Extraction, Community Detection, God Nodes (+5 more)

### Community 84 - "Update Command"
Cohesion: 0.15
Nodes (5): check_outdated(), init_cmd(), _load_model_string(), sessions_cmd(), update_cmd()

### Community 90 - "CLI Tools Init"
Cohesion: 0.18
Nodes (4): _load_project_tool_agent_config(), load_tools_from_settings(), _parse_agent_target(), chat_cmd()

### Community 91 - "Kader Config"
Cohesion: 0.17
Nodes (6): ensure_env_file(), ensure_kader_directory(), _ensure_settings_file(), get_kader_directory(), initialize_kader_config(), load_env_file()

### Community 95 - "CLI Design HTML"
Cohesion: 0.20
Nodes (11): Plan Sidebar, Kader CLI Prompt Mode Mockup, Kader CLI Prompt Mode Mockup (Alt), Kader CLI System Mode Mockup, Kader CLI Command Mode Mockup, CLI Session Management, CLI Reference, Connect Command (+3 more)

### Community 96 - "CLI Screenshots"
Cohesion: 0.25
Nodes (11): ATIS / SNIPS Training Datasets, Chat Conversation Interface, diet_data Dataset, Provider Connection Error Handling, Files Panel, Kader CLI v1.2.1 Interface, Ollama LLM Provider (ollama:kimi-k2.5:cloud), Plan / Task Panel (+3 more)

### Community 99 - "Documentation Agents"
Cohesion: 0.25
Nodes (8): BaseAgent, PlanningAgent, ReActAgent, Agent YAML Configuration, Agents Concept, Core Framework, Tools Concept, Workflows

### Community 104 - "Settings Migration"
Cohesion: 0.27
Nodes (3): _migrate_user_subagents(), create_subagent_dir_yaml(), TestKaderSettingsSubagentsMigration

### Community 112 - "Graphify Transcribe"
Cohesion: 0.22
Nodes (6): GRAPHIFY_WHISPER_PROMPT, transcribe_all, build_merge, Cluster-Only Mode (--cluster-only), detect_incremental, Extraction Manifest

### Community 114 - "Callbacks Documentation"
Cohesion: 0.28
Nodes (7): BaseCallback, CallbackContext, CallbackEvent, LLMCallback, LoggingLLMCallback, LoggingToolCallback, ToolCallback

### Community 115 - "Memory Documentation"
Cohesion: 0.22
Nodes (9): Memory Concept, AgentState, Checkpointer, ContextAggregator, FileSessionManager, PersistentSlidingWindowConversationManager, SlidingWindowConversationManager, ToolOutputCompressor (+1 more)

### Community 116 - "Python Developer Example"
Cohesion: 0.33
Nodes (3): main(), PlanningAgent, ReActAgent

### Community 119 - "Contributing Guide"
Cohesion: 0.25
Nodes (7): Build/Lint/Test Commands, Kader Code Style Guidelines, Kader Agent Instructions, Contributing Code Style Guidelines, Contributing to Kader, Development Setup, Pull Request Process

### Community 130 - "Graphify Query"
Cohesion: 0.47
Nodes (4): graphify query CLI, LESSONS.md, save-result, Graph Traversal (BFS/DFS)

### Community 131 - "CLI Screenshots"
Cohesion: 0.33
Nodes (6): Chat Interface, Command Palette, Keyboard Shortcuts Bar, Ollama Connection Error, Plan Panel, Shell Execution Mode

### Community 133 - "CLI Image Docs"
Cohesion: 0.33
Nodes (6): Kader CLI Application, Help Command /help, Kader ASCII Art Logo, Model Display minimax-m2.5:cloud, Input Prompt [>], Version Display v2.7.0

### Community 135 - "Configuration Docs"
Cohesion: 0.40
Nodes (5): Kader Configuration, Environment Variables, Kader Directory (~/.kader), Kader Settings, OllamaProvider

### Community 136 - "Providers Documentation"
Cohesion: 0.40
Nodes (6): Providers Concept, AnthropicProvider, GoogleProvider, Message Helper, OpenAICompatibleProvider, OpenAIProviderConfig

### Community 150 - "Documentation Agents"
Cohesion: 0.40
Nodes (5): AgentTool, ReActAgent, AgentTool, BaseAgent, ReActAgent

### Community 154 - "Provider README"
Cohesion: 0.40
Nodes (5): AnthropicProvider, GoogleProvider, MistralProvider, OllamaProvider, OpenAICompatibleProvider

### Community 169 - "Memory README"
Cohesion: 0.67
Nodes (3): AgentState, FileSessionManager, SlidingWindowConversationManager

### Community 170 - "Callbacks README"
Cohesion: 0.67
Nodes (3): BaseCallback, LLMCallback, ToolCallback

## Knowledge Gaps
- **106 isolated node(s):** `$schema`, `plugin`, `kader`, `opencode Workflow`, `Contributing to Kader Skill (.opencode)` (+101 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1812 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **137 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Message` connect `Anthropic Provider Examples` to `LLM Provider Base`, `Agent Examples`, `Memory & Conversation`, `OpenAI Compatible Provider`, `Mistral Provider Examples`, `Google Provider Examples`, `Ollama Provider Examples`, `Provider Demo Patterns`, `CLI Callbacks & Sessions`, `Configuration & Google GenAI`, `Anthropic Provider Impl`, `Base Agent Core`, `Provider Base Tests`, `Session Save/Load`, `Provider Base`, `Test Infrastructure`, `Agent Tool`, `Memory Example`, `Google Provider`, `Kader Init & Checkpointer`, `Planner Executor Workflow`, `Context Aggregator`, `OpenAI Compatible Provider`, `Agent Tool Execution`?**
  _High betweenness centrality (0.145) - this node is a cross-community bridge._
- **Why does `AgentTool` connect `Agent Tool` to `Agent Examples`, `Memory & Conversation`, `Agent Tool Tests`, `Anthropic Provider Examples`, `Configuration & Google GenAI`, `Agent Tool Tests`, `Tool Base Patterns`, `CLI App Main`, `Test Infrastructure`, `Memory Compression`, `Agent Tool Execute`, `Skills Tool`, `Hello Skill Example`, `Agent Tool Skills Tests`, `CLI App Chat`, `Tool Execution`, `Agent Tool Skills Tests`, `Update Command`, `Agent Tool Config`, `Agent Tool Tests`, `Initialize Command`, `Persistent Conversation`, `Subagent Workflows`, `Python Developer Example`, `Agent Tool Execution`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Why does `ModelConfig` connect `Anthropic Provider Impl` to `LLM Provider Base`, `Memory & Conversation`, `OpenAI Compatible Provider`, `Anthropic Provider Examples`, `Mistral Provider Examples`, `Google Provider Examples`, `Ollama Provider Examples`, `Provider Demo Patterns`, `Google Provider`, `CLI Callbacks & Sessions`, `Configuration & Google GenAI`, `Base Agent Core`, `LLM Provider Factory`, `OpenAI Compatible Provider`, `Provider Base`, `OpenAI Compatible Impl`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Are the 75 inferred relationships involving `Message` (e.g. with `demo_async_invocation()` and `demo_async_streaming()`) actually correct?**
  _`Message` has 75 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `ModelConfig` (e.g. with `BaseAgent` and `HierarchicalConversationManager`) actually correct?**
  _`ModelConfig` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `OpenAICompatibleProvider` (e.g. with `demo_list_models()` and `LLMProviderFactory`) actually correct?**
  _`OpenAICompatibleProvider` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `Usage` (e.g. with `BaseAgent` and `AnthropicProvider`) actually correct?**
  _`Usage` has 20 INFERRED edges - model-reasoned connections that need verification._