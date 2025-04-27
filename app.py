from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from faicons import icon_svg

# Load the dataset
data = pd.read_csv("garments_worker_productivity.csv")

# Data transformations 
data['idle_men'] = data['idle_men'].apply(lambda x: 'No idle men' if x == 0 else 'Has idle men')
data['idle_time'] = data['idle_time'].apply(lambda x: 'No idle time' if x == 0 else 'Has idle time')
data['team'] = pd.qcut(data['team'], q=4, labels=['Team Group 1', 'Team Group 2', 'Team Group 3', 'Team Group 4']).astype(str)
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data.fillna(data['wip'].mean(), inplace = True)

# Strip spaces and fix typos
data['department'] = data['department'].str.strip().replace({
    'finishing ': 'finishing',
    'finishing': 'finishing',  #just to be safe
    'sweing': 'sewing'
})

num_cols = data.select_dtypes(include='number').columns.tolist()

app_ui = ui.page_navbar(
    
    # Landing Page
    ui.nav_panel(
        "Landing Page",
        ui.input_dark_mode(),
        ui.layout_columns(
            ui.value_box("Total Average Productivity", ui.output_text("kpi_actual_prod"), showcase=icon_svg("chart-line"),  theme="gradient-blue-indigo"),
            ui.value_box("Productivity per Worker", ui.output_text("kpi_per_worker"), showcase=icon_svg("person"), theme="gradient-blue-indigo"),
            ui.value_box("Overtime Incentive Ratio", ui.output_text("kpi_ot_ratio"), showcase=icon_svg("clock"), theme="gradient-blue-indigo"),
            ui.value_box("RMSE (Actual vs Targeted)", ui.output_text("kpi_rmse"), showcase=icon_svg("chart-bar"), theme="gradient-blue-indigo")
        ),
        ui.row(
            ui.h4("Productivity Analysis")
        ),
        ui.row(
            ui.card(output_widget("ridge_plot"))
            
        ),
        ui.card(output_widget("data_dictionary"))

        
    ),

    # Page 1: Distributions and Bar Charts
    ui.nav_panel(
        "Distributions Overview",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_date_range(
                    "date_range", "Date Range Slicer:",
                    start=data["date"].min(),
                    end=data["date"].min()
                ),
                ui.input_select("categorical_col", "Select Category:", 
                                choices=["department", "day", "quarter", "team"], selected="department")
            ),
            ui.layout_columns(
                ui.card(output_widget("productivity_hist")),
                ui.card(output_widget("categorical_bar"))
            )
        ),
        ui.row(
            ui.card(ui.output_text_verbatim("summary_text"))
        )
    ),

    # Page 2: Actual vs Expected Productivity by Category
    ui.nav_panel(
        "Productivity by Category and Day",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select("category_selector", "Select Categorical Variable:", 
                                choices=["department", "team", "quarter"], selected="department"),
                ui.input_radio_buttons("prod_type", "Select Productivity Type:",
                                    choices=["actual_productivity", "targeted_productivity"],
                                    selected="actual_productivity")
            ),
            ui.layout_columns(
                ui.card(output_widget("box_plot_prod"))
            )
        ),
        ui.row(
            ui.card(output_widget("summary_table"))
        )
    ),


    # Page 3: Custom Scatter Plot
    ui.nav_panel(
        "Custom Productivity Scatter",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select("y_productivity", "Y-axis Variable:",
                                choices=["actual_productivity", "targeted_productivity"], selected="actual_productivity"),
                ui.input_select("x_numeric", "X-axis Variable:", choices=num_cols, selected="over_time"),
                ui.input_select("color_by", "Color By:",
                                choices=["department", "team", "quarter", "day"], selected="team")
            ),
            ui.card(output_widget("custom_scatter"))
        ),
        ui.row(
            ui.card(ui.output_text_verbatim("scatter_text"))
        )
    ),


    # Page 4: Time Series Visualization
    ui.nav_panel(
        "Productivity Over Time",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_selectize("ts_y_vars", "Select Numerical Variable(s):",
                                choices=num_cols, multiple=True, selected=["actual_productivity"]),
                ui.input_selectize("ts_departments", "Select Departments:",
                                choices=sorted(data["department"].unique()), multiple=True,
                                selected=sorted(data["department"].unique())[:2]),
                ui.input_date_range("ts_date_range", "Select Date Range:",
                                    start=data["date"].min(), end="2015-02-01")
            ),
            ui.card(output_widget("ts_plot"))
        ),
        ui.row(
            ui.card(ui.output_text_verbatim("time_text"))
        )
    ),

    # Page 5: PCA and KMeans Clustering
    ui.nav_panel(
        "PCA Clustering",
        ui.row(
            ui.column(6, ui.card(output_widget("pca_kmeans_plot"))),
            ui.column(6, ui.card(output_widget("pca_table")))
        ),
        ui.row(
            ui.card(ui.output_text_verbatim("pca_text"))
        )
    ),

    # Page 6: Dataset Viewer
    ui.nav_panel(
        "View Dataset",
            ui.output_data_frame("data_table")
    )
)

def server(input, output, session):

    # Landing Page 
    # KPI Cards
    @output
    @render.text
    def kpi_actual_prod():
        return f"{data['actual_productivity'].mean():.2f}"

    @output
    @render.text
    def kpi_per_worker():
        data['productivity_per_worker'] = data['actual_productivity'] / data['no_of_workers']
        return f"{data['productivity_per_worker'].mean():.2f}"

    @output
    @render.text
    def kpi_ot_ratio():
        data['over_time_ratio'] = data['over_time'] / (data['over_time'] + 1 + data['incentive'])
        return f"{data['over_time_ratio'].mean():.2f}"
 
    @output
    @render.text
    def kpi_rmse():
        rmse = mean_squared_error(data['targeted_productivity'], data['actual_productivity'])
        return f"{rmse:.2f}"

    @output
    @render.text
    def kpi_idle():
        return f"{data[data['idle_men'] == 'Has idle men']['actual_productivity'].mean():.2f}"

    @output
    @render.text
    def kpi_non_idle():
        return f"{data[data['idle_men'] == 'No idle men']['actual_productivity'].mean():.2f}"

    @output
    @render.text
    def kpi_total_workers():
        return f"{data['no_of_workers'].sum()}"

    # Variable Description Table
    @output
    @render_widget
    def data_dictionary():
        import plotly.figure_factory as ff

        var_description = {
            "date": "Date of observation",
            "day": "Day of the week",
            "quarter": "Quarter of the month",
            "department": "Department (sewing or finishing)",
            "team": "Team grouping",
            "no_of_workers": "Number of workers involved",
            "smv": "Standard Minute Value, it is the allocated time for a task",
            "wip": "Work in progress. Includes the number of unfinished items for products",
            "over_time": "Overtime minutes",
            "incentive": "Financial incentives provided",
            "idle_time": "Idle time presence",
            "idle_men": "Idle men presence",
            "actual_productivity": "Measured actual productivity",
            "targeted_productivity": "Targeted/expected productivity",
            "productivity_per_worker": "Actual productivity divided by workers",
            "over_time_ratio": "Overtime ratio calculated"
        }

        dict_df = pd.DataFrame(list(var_description.items()), columns=["Variable", "Description"])
        fig = ff.create_table(dict_df)
        return fig

    # Ridge Plot
    @output
    @render_widget
    def ridge_plot():
        import plotly.figure_factory as ff
        grouped = [data[data['day'] == day]['actual_productivity'].values for day in data['day'].unique()]
        fig = ff.create_distplot(grouped, group_labels=data['day'].unique().tolist(), show_hist=False, show_rug=False)
        fig.update_layout(
            title="Distribution of Productivity by Day (Ridge Plot)",
            plot_bgcolor='white',
            margin=dict(t=60, l=40, r=40, b=60),
            font=dict(family="Arial", size=10)
        )
        return fig

    # Page 1
    @output
    @render_widget
    def productivity_hist():
        start, end = input.date_range()

        df = data[(data["date"] >= pd.to_datetime(start)) & (data["date"] <= pd.to_datetime(end))]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df["actual_productivity"], name="Actual Productivity", marker_color="#1f77b4"))
        fig.add_trace(go.Histogram(x=df["targeted_productivity"], name="Targeted Productivity", marker_color="#ff7f0e"))

        fig.update_layout(
            barmode='overlay',
            title={"text": "Distribution of Actual vs Targeted Productivity", "y": 0.93, "x": 0.5, "xanchor": "center", "yanchor": "top"},
            xaxis_title="Productivity",
            yaxis_title="Frequency",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.08,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0)'
            ),
            plot_bgcolor='white',
            margin=dict(t=80, l=40, r=40, b=60),
            font=dict(family="Arial", size=10),
        )
        fig.update_traces(opacity=0.6)
        fig.update_xaxes(showgrid=True, gridcolor="lightgrey")
        fig.update_yaxes(showgrid=True, gridcolor="lightgrey")

        return fig

    @output
    @render_widget
    def categorical_bar():
        start, end = input.date_range()
        df = data[(data["date"] >= pd.to_datetime(start)) & (data["date"] <= pd.to_datetime(end))]

        col = input.categorical_col()

        # Apply day orders
        if col == "day":
            ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            df[col] = pd.Categorical(df[col], categories=ordered_days, ordered=True)
        
        count_df = df.groupby(col).size().reset_index(name="Count")


        fig = px.bar(
            count_df, x=col, y="Count", color=col,
            template="simple_white",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig.update_layout(
            barmode='overlay',
            title={"text": "Distribution of Productivity by Category", "y": 0.93, "x": 0.5, "xanchor": "center", "yanchor": "top"},
            yaxis_title="Frequency",
            # legend=dict(
            #     orientation="h",
            #     yanchor="top",
            #     y=1.08,
            #     xanchor="center",
            #     x=0.5,
            #     bgcolor='rgba(0,0,0,0)'
            # ),
            plot_bgcolor='white',
            margin=dict(t=80, l=40, r=40, b=60),
            font=dict(family="Arial", size=10),
        )
        fig.update_traces(opacity=0.8)
        fig.update_xaxes(showgrid=True, gridcolor="lightgrey")
        fig.update_yaxes(showgrid=True, gridcolor="lightgrey")

        return fig
    
    @output
    @render.text
    def summary_text():
        return "You are viewing productivity distribution and categorical breakdown. Use the date range slicer and dropdown menu to explore specific timeframes and groupings."


    #Page 2
    @output
    @render_widget
    def box_plot_prod():
        cat = input.category_selector()
        prod = input.prod_type()

        fig = px.box(
            data, x="day", y=prod, color=cat,
            title=f"{prod.replace('_', ' ').title()} by Day and {cat.capitalize()}",
            template="simple_white", 
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_traces(quartilemethod="inclusive")
        return fig



    @output
    @render_widget
    def summary_table():
        import plotly.figure_factory as ff

        cat = input.category_selector()
        grouped = data.groupby([cat, "day"])[["actual_productivity", "targeted_productivity"]].mean().reset_index()
        table_fig = ff.create_table(grouped.round(2))
        return table_fig

    # Page 3
    @output
    @render_widget
    def custom_scatter():
        x = input.x_numeric()
        y = input.y_productivity()
        color = input.color_by()

        fig = px.scatter(data, x=x, y=y, color=data[color],
                         hover_name="department", title=f"{y.replace('_', ' ').title()} vs {x.replace('_', ' ').title()}",
                         size="no_of_workers", log_x=True)
        return fig

    @output
    @render.text
    def scatter_text():
        return f"You are viewing a scatter plot of {input.y_productivity().replace('_', ' ')} vs {input.x_numeric().replace('_', ' ')} colored by {input.color_by()} and sized by number of workers."

    # Page 4
    @output
    @render_widget
    def ts_plot():
        df = data.copy()

        start_date, end_date = input.ts_date_range()
        y_vars = input.ts_y_vars()
        depts = input.ts_departments()

        if start_date and end_date:
            df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

        if depts:
            df = df[df["department"].isin(depts)]

        df["date"] = pd.to_datetime(df["date"])

        if df.empty or not y_vars:
            fig = go.Figure()
            fig.update_layout(title="No data available for the selected filters.",
                            template="simple_white")
            return fig

        fig = go.Figure()
        for y in y_vars:
            for dept in df["department"].unique():
                filtered = df[df["department"] == dept]
                fig.add_trace(go.Scatter(
                    x=filtered["date"], y=filtered[y],
                    mode="lines",
                    name=f"{dept} - {y}",
                    line=dict(width=2)
                ))

        fig.update_layout(
            title={"text": "Time Series of Selected Variables", "x": 0.5},
            template="simple_white",
            xaxis_title="Date",
            yaxis_title="Value",
            xaxis_tickformat="%Y-%m-%d",
            margin=dict(t=60, l=40, r=40, b=60),
            font=dict(family="Arial", size=10),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.3,
                xanchor="center", x=0.5
            )
        )

        return fig
    
    @output
    @render.text
    def time_text():
        return f"You are viewing a time series plot of the selected Category vs the selected Time Range colored by Department."


    #Page 5
    @reactive.Calc
    def pca_results():
        df = data[num_cols].dropna()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled)

        cluster = KMeans(n_clusters=2, random_state=42).fit_predict(scaled)

        pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
        pca_df["Cluster"] = cluster
        pca_df["Department"] = data.loc[df.index, "department"].values

        components_df = pd.DataFrame(pca.components_.T, index=num_cols, columns=["PC1", "PC2"])

        return {"pca_df": pca_df, "components": components_df}

    @output
    @render_widget
    def pca_kmeans_plot():
        df = pca_results()["pca_df"]
        df["Cluster"] = df["Cluster"].astype(str)
        fig = px.scatter(df, x="PC1", y="PC2", color="Cluster",
                        title="KMeans Clustering Visualized on PCA")

        fig.update_traces(marker=dict(showscale=False))

        return fig

    @output
    @render_widget
    def pca_table():
        from plotly.figure_factory import create_table
        table_df = pca_results()["components"].round(3).reset_index().rename(columns={"index": "Feature"})
        fig = create_table(table_df)
        return fig

    @output
    @render.text
    def pca_text():
        return "You are viewing a PCA plot that shows clusters based on the main variance-driving components in the dataset."


    #Page 6
    @output
    @render.data_frame
    def data_table():
        return render.DataTable(data, filters = True)



app = App(app_ui, server)