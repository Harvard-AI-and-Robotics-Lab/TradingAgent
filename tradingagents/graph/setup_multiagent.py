# TradingAgents/graph/setup.py

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.agent_utils import Toolkit

from .conditional_logic import ConditionalLogic


class GraphSetupMultiAgent:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        config: Dict[str, Any],
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        analyst_llms: List[ChatOpenAI],
        credibility_scorer_llm: ChatOpenAI,
        toolkit: Toolkit,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory=None,
        conditional_logic: ConditionalLogic = None,
        skip_analysts: bool = False,
    ):
        """Initialize with required components."""
        self.config = config
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.analyst_llms = analyst_llms if analyst_llms is not None and len(analyst_llms) > 0 else [quick_thinking_llm]
        self.credibility_scorer_llm = credibility_scorer_llm if credibility_scorer_llm is not None else deep_thinking_llm
        self.toolkit = toolkit
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.conditional_logic = conditional_logic
        self.skip_analysts = skip_analysts

    def setup_graph(
        self, selected_analysts=["price", "market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "price": Price analyst
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Create analyst nodes
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}
        credibility_scorer_nodes = {}
        credibility_scorer_delete_nodes = {}

        suffix_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
        if "price" in selected_analysts:
            for i, analyst_llm in enumerate(self.analyst_llms):
                i_suffix = suffix_map[i]
                analyst_nodes[f"price_{i_suffix}"] = create_price_analyst(
                    analyst_llm, self.toolkit
                )
                delete_nodes[f"price_{i_suffix}"] = create_msg_delete()
                tool_nodes[f"price_{i_suffix}"] = self.tool_nodes["price"]

            credibility_scorer_nodes[f"price"] = create_price_credibility_scorer(
                self.credibility_scorer_llm, self.toolkit
            )
            credibility_scorer_delete_nodes[f"price"] = create_msg_delete()

        if "fundamentals" in selected_analysts:
            if not self.skip_analysts:
                for i, analyst_llm in enumerate(self.analyst_llms):
                    i_suffix = suffix_map[i]
                    analyst_nodes[f"fundamentals_{i_suffix}"] = create_fundamentals_analyst(
                        analyst_llm, self.toolkit
                    )
                    delete_nodes[f"fundamentals_{i_suffix}"] = create_msg_delete()
                    tool_nodes[f"fundamentals_{i_suffix}"] = self.tool_nodes["fundamentals"]

            credibility_scorer_nodes[f"fundamentals"] = create_fundamentals_credibility_scorer(
                self.credibility_scorer_llm, self.toolkit
            )
            credibility_scorer_delete_nodes[f"fundamentals"] = create_msg_delete()

        if "market" in selected_analysts:
            if not self.skip_analysts:
                # y_finance data
                for i, analyst_llm in enumerate(self.analyst_llms):
                    i_suffix = suffix_map[i]
                    analyst_nodes[f"market_{i_suffix}"] = create_market_analyst(
                        analyst_llm, self.toolkit
                    )
                    delete_nodes[f"market_{i_suffix}"] = create_msg_delete()
                    tool_nodes[f"market_{i_suffix}"] = self.tool_nodes["market"]

            credibility_scorer_nodes[f"market"] = create_market_credibility_scorer(
                self.credibility_scorer_llm, self.toolkit
            )
            credibility_scorer_delete_nodes[f"market"] = create_msg_delete()

        if "social" in selected_analysts:
            if not self.skip_analysts:
                for i, analyst_llm in enumerate(self.analyst_llms):
                    i_suffix = suffix_map[i]
                    analyst_nodes[f"social_{i_suffix}"] = create_social_media_analyst(
                        analyst_llm, self.toolkit
                    )
                    delete_nodes[f"social_{i_suffix}"] = create_msg_delete()
                    tool_nodes[f"social_{i_suffix}"] = self.tool_nodes["social"]

            credibility_scorer_nodes[f"social"] = create_social_media_credibility_scorer(
                self.credibility_scorer_llm, self.toolkit
            )
            credibility_scorer_delete_nodes[f"social"] = create_msg_delete()

        if "news" in selected_analysts:
            if not self.skip_analysts:
                for i, analyst_llm in enumerate(self.analyst_llms):
                    i_suffix = suffix_map[i]
                    analyst_nodes[f"news_{i_suffix}"] = create_news_analyst(
                        analyst_llm, self.toolkit
                    )
                    delete_nodes[f"news_{i_suffix}"] = create_msg_delete()
                    tool_nodes[f"news_{i_suffix}"] = self.tool_nodes["news"]

            credibility_scorer_nodes[f"news"] = create_news_credibility_scorer(
                self.credibility_scorer_llm, self.toolkit
            )
            credibility_scorer_delete_nodes[f"news"] = create_msg_delete()

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(
            self.quick_thinking_llm, self.bull_memory
        )
        bear_researcher_node = create_bear_researcher(
            self.quick_thinking_llm, self.bear_memory
        )
        research_manager_node = create_research_manager(
            self.deep_thinking_llm, self.invest_judge_memory
        )
        trader_node = create_trader(
            self.quick_thinking_llm, self.trader_memory
        )

        # Create risk analysis nodes
        if self.risk_manager_memory is not None:
            risky_analyst = create_risky_debator(self.quick_thinking_llm)
            neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
            safe_analyst = create_safe_debator(self.quick_thinking_llm)
            risk_manager_node = create_risk_manager(
                self.deep_thinking_llm, self.risk_manager_memory
            )

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Add credibility scorer nodes to the graph
        for analyst_type, node in credibility_scorer_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Credibility Scorer", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()} Credibility Scorer", credibility_scorer_delete_nodes[analyst_type]
            )

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        if self.risk_manager_memory is not None:
            workflow.add_node("Risky Analyst", risky_analyst)
            workflow.add_node("Neutral Analyst", neutral_analyst)
            workflow.add_node("Safe Analyst", safe_analyst)
            workflow.add_node("Risk Judge", risk_manager_node)

        # Define edges
        if not self.skip_analysts:
            # Start with the first analyst
            first_analyst = selected_analysts[0] + '_a'
            workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

            # Connect analysts in sequence
            for i, analyst_type_ in enumerate(selected_analysts):

                for j, analyst_llm in enumerate(self.analyst_llms):
                    j_suffix = suffix_map[j]
                    analyst_type = analyst_type_ + '_' + j_suffix
                    current_analyst = f"{analyst_type.capitalize()} Analyst"
                    current_tools = f"tools_{analyst_type}"
                    current_clear = f"Msg Clear {analyst_type.capitalize()}"

                    # Add conditional edges for current analyst
                    workflow.add_conditional_edges(
                        current_analyst,
                        lambda state, analyst_type=analyst_type: self.conditional_logic.should_continue(state, analyst_type),
                        [current_tools, current_clear],
                    )
                    workflow.add_edge(current_tools, current_analyst)

                    if j < len(self.analyst_llms) - 1:
                        next_analyst = f"{analyst_type_.capitalize()}_{suffix_map[j+1]} Analyst"
                        workflow.add_edge(current_clear, next_analyst)
                    else:
                        # Add credibility scorer edges for current analyst
                        current_credibility_scorer = f"{analyst_type_.capitalize()} Credibility Scorer"
                        workflow.add_edge(current_clear, current_credibility_scorer)
                        current_credibility_scorer_clear = f"Msg Clear {analyst_type_.capitalize()} Credibility Scorer"
                        workflow.add_edge(current_credibility_scorer, current_credibility_scorer_clear)

                        # Connect to next analyst or to Bull Researcher if this is the last analyst
                        if i < len(selected_analysts) - 1:
                            next_analyst = f"{selected_analysts[i+1].capitalize()}_a Analyst"
                            workflow.add_edge(current_credibility_scorer_clear, next_analyst)
                        else:
                            workflow.add_edge(current_credibility_scorer_clear, "Bull Researcher")

        else:
            if "price" in selected_analysts:
                analyst_type_ = "price"
                workflow.add_edge(START, f"{analyst_type_.capitalize()}_a Analyst")

                for j, analyst_llm in enumerate(self.analyst_llms):
                    j_suffix = suffix_map[j]
                    analyst_type = analyst_type_ + '_' + j_suffix
                    current_analyst = f"{analyst_type.capitalize()} Analyst"
                    current_tools = f"tools_{analyst_type}"
                    current_clear = f"Msg Clear {analyst_type.capitalize()}"

                    # Add conditional edges for current analyst
                    workflow.add_conditional_edges(
                        current_analyst,
                        lambda state, analyst_type=analyst_type: self.conditional_logic.should_continue(state, analyst_type),
                        [current_tools, current_clear],
                    )
                    workflow.add_edge(current_tools, current_analyst)

                    if j < len(self.analyst_llms) - 1:
                        next_analyst = f"{analyst_type_.capitalize()}_{suffix_map[j+1]} Analyst"
                        workflow.add_edge(current_clear, next_analyst)
                    else:
                        # Add credibility scorer edges for current analyst
                        current_credibility_scorer = f"{analyst_type_.capitalize()} Credibility Scorer"
                        workflow.add_edge(current_clear, current_credibility_scorer)
                        current_credibility_scorer_clear = f"Msg Clear {analyst_type_.capitalize()} Credibility Scorer"
                        workflow.add_edge(current_credibility_scorer, current_credibility_scorer_clear)

                        # Connect to next analyst or to Bull Researcher if this is the last analyst
                        first_scorer = None
                        for analyst_type_ in selected_analysts:
                            if analyst_type_ != "price":
                                first_scorer = analyst_type_
                                break
                        if first_scorer is not None:
                            first_scorer = f"{first_scorer.capitalize()} Credibility Scorer"
                            workflow.add_edge(current_credibility_scorer_clear, first_scorer)
                        else:
                            workflow.add_edge(current_credibility_scorer_clear, "Bull Researcher")

            else:
                # Start with the first scorer
                for analyst_type_ in selected_analysts:
                    if analyst_type_ != "price":
                        first_scorer = analyst_type_
                        break
                first_scorer = selected_analysts[0]
                workflow.add_edge(START, f"{first_scorer.capitalize()} Credibility Scorer")

            for i, analyst_type_ in enumerate(selected_analysts):
                if analyst_type_ == "price":
                    continue
                # Add credibility scorer edges for current analyst
                current_credibility_scorer = f"{analyst_type_.capitalize()} Credibility Scorer"
                current_credibility_scorer_clear = f"Msg Clear {analyst_type_.capitalize()} Credibility Scorer"
                workflow.add_edge(current_credibility_scorer, current_credibility_scorer_clear)

                if i < len(selected_analysts) - 1:
                    next_credibility_scorer = f"{selected_analysts[i+1].capitalize()} Credibility Scorer"
                    workflow.add_edge(current_credibility_scorer_clear, next_credibility_scorer)
                else:
                    workflow.add_edge(current_credibility_scorer_clear, "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")

        if self.risk_manager_memory is None:
            workflow.add_edge("Trader", END)

        else:
            workflow.add_edge("Trader", "Risky Analyst")

            workflow.add_conditional_edges(
                "Risky Analyst",
                self.conditional_logic.should_continue_risk_analysis,
                {
                    "Safe Analyst": "Safe Analyst",
                    "Risk Judge": "Risk Judge",
                },
            )
            workflow.add_conditional_edges(
                "Safe Analyst",
                self.conditional_logic.should_continue_risk_analysis,
                {
                    "Neutral Analyst": "Neutral Analyst",
                    "Risk Judge": "Risk Judge",
                },
            )
            workflow.add_conditional_edges(
                "Neutral Analyst",
                self.conditional_logic.should_continue_risk_analysis,
                {
                    "Risky Analyst": "Risky Analyst",
                    "Risk Judge": "Risk Judge",
                },
            )

            workflow.add_edge("Risk Judge", END)

        # Compile and return
        return workflow.compile()