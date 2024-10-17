import streamlit as st

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#read df file
df=pd.read_csv("whp_full.csv")

#create a streamlit shell
st.title("World Hapiness Report")
st.sidebar.title("Navigation")
pages=["Introduction", "Data Exploration", "Data Visualization", "Modelling", "Prediction", "Conclusion" ]
page=st.sidebar.radio("Go to", pages)

#On the first page (page [0])
if page == pages[0]: 
  st.header("Introduction") #create a header
  st.info("This study aims to leverage machine learning for analyzing historical world hapiness rate data and forecasting world hapiness with machine learning.")
  st.write("### Objectives")
  st.write("- Analyze world hapiness data for different years to identify global patterns")
  st.write("- Investigate relationships between happiness and factors like GDP, Social Support etc.")
  st.write("- Develop predictive models for forecasting world hapiness")

  st.markdown("""
  <style>
		.right-align {
			text-align: right;
		}
	</style>
	""", unsafe_allow_html=True)

  st.markdown('<p class="right-align">Data Team: Tetiana Dul, Ismail Ocakdan, Felipe Forero</p>', unsafe_allow_html=True)
  st.markdown('<p class="right-align">Date: October 28, 2024</p>', unsafe_allow_html=True)




#On the second page (page [1])
if page == pages[1]:
    st.header("Data Exploration")
    st.subheader("Data Sources")
    """This project utilizes two main datasets:"""
    #load whp 2021 dataset
    whp_2021_report = pd.read_csv("world-happiness-report-2021.csv")
    with st.expander("**World Hapiness Report 2021**"):
        """
        - Contains data on World Hapiness in 2021, GDP, Social Support, Healthy life expectancy and other related factors across countries in 2021.
        - Has 149 rows and 20 columns.
        - Freely available online.
        """
        st.write("Preview of the World Hapiness Report 2021 File:")
        st.dataframe(whp_2021_report.head())
        
        """- The dataset is very clean and doesn't have any missing values."""
    #load whp_historical dataset
    whp_historical = pd.read_csv("world-happiness-report.csv")
    with st.expander("**World Hapiness Report Historical Data**"):
        """
        - Contains a historical data of happiness score (Ladder Score)of different countries over 15 years.
        - Has 1949 rows and 9 columns.
        - Freely available online.
		"""
        st.write("Preview of the World Hapiness Report File:")
        st.dataframe(whp_historical.head())
        """- The dataset contains about 1,75% of missing values."""
    st.subheader("Data Merging and cleaning")
    """
    - Rename needed columns and bring both datasets to the same shape.
    - Combine 2021 report with historical data dataset.
    - Clean the data by removing rows with missing values.
    """
    

    whp_full = pd.read_csv("whp_full.csv")
    with st.expander("**Cleaned Data**"):
        st.write("Preview of the Cleaned Data:")
        st.dataframe(whp_full.head())

        "Exploring variables"
        """
        **Ladder Score:**
        Hapiness score, the target variable in our report.

        **Logged GDP per capita:**
        The natural logarithm of a country’s GDP per capita, used to represent the economic prosperity of a country.
        """

        """
        **Social support:**
        A measure of the strength of social networks and the availability of assistance in times of need.
        
        **Healthy life expectancy:**
        The average number of years a person can expect to live in good health in a given country.
        """

        """
        **Freedom to make life choices:**
         The degree to which people feel free to make decisions about their own lives.
         
         **Generosity:**
         A measure of people's willingness to donate money or time to others.
        """

        """
        **Perceptions of corruption:**
        The public's perception of corruption within government and business in a given country.
        """


#page number 3 ([2])
if page == pages[2]:

    def create_plot(year):
        # Filter data for the selected year
        df_year = df[df['year'] == year]
        # Get top 10 and bottom 10 countries
        top_10 = df_year.nlargest(10, 'Ladder score')
        bottom_10 = df_year.nsmallest(10, 'Ladder score').sort_values('Ladder score', ascending=False)
        #Combine datasets
        plot_data = pd.concat([top_10, bottom_10])
        # Create category colum
        plot_data['Category'] = ['Top 10' if x in top_10.index else 'Bottom 10' for x in plot_data.index]
        #Create the plot
        fig = px.bar(plot_data, 
            x='Country name',  
            y='Ladder score',
            color='Category',
            title=f'10 countries with the highest and lowest Ladder score in {year}',
            labels={'Country name': 'Country', 'Ladder score': 'Happiness Score'},
            color_discrete_map={'Top 10': 'green', 'Bottom 10': 'red'},
            height=600,
            text='Ladder score')
        fig.update_layout(xaxis_tickangle=-45)
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        return fig

    st.subheader('World Happiness over the years')
    # Create year slider
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    selected_year = st.slider('Select a year', min_year, max_year, max_year)

    # Create and display the plot
    fig = create_plot(selected_year)
    st.plotly_chart(fig, use_container_width=True)

    #a scatter plot of hapiness scores by regions in 2021
    st.subheader('Hapiness scores in different regions in 2021')
    import plotly.express as px

    
    whp_2021_report = pd.read_csv("world-happiness-report-2021.csv")
    fig = px.scatter(
        whp_2021_report,
        x="Regional indicator",
        y="Ladder score",
        hover_data=["Country name"],
        color="Regional indicator",
        labels={
            "Regional indicator": "",
            "Ladder score": "Ladder score",
            "Country name": "Country"
        },
        height=600
    )

    # Customize the layout
    fig.update_layout(
        title={
            'text': 'Happiness by regions',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=False,
        xaxis_tickangle=-45,
        plot_bgcolor='white'
    )

    # Add gridlines
    fig.update_yaxes(
        gridcolor='lightgrey',
        gridwidth=0.5,
        range=[2, 8]
    )
    
    st.plotly_chart(fig, use_container_width=True)

    #barplot of an average hapiness scores by region
    st.subheader('Average hapiness score by regions in 2021') 
    
    # Calculate average ladder score by region
    region_avg = (whp_2021_report.groupby('Regional indicator')['Ladder score']
              .mean()
              .sort_values(ascending=False)
              .reset_index())

    # Create the bar plot
    fig = px.bar(
        region_avg,
        x='Regional indicator',
        y='Ladder score',
        labels={
            'Regional indicator': '',
            'Ladder score': 'Average Ladder Score'
        },
        title='Average Happiness Score by Region',
        text='Ladder score'  # This adds the numbers on top of the bars
    )

    # Customize the layout
    fig.update_layout(
        title={
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95
        },
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        showlegend=False,
        height=600
    )

    # Add gridlines
    fig.update_yaxes(
        gridcolor='lightgrey',
        gridwidth=0.5
    )
    # add text position to be above bars
    fig.update_traces(textposition='outside')
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)


    # Add a histogram of hapiness score distributions acrosscountries
    st.subheader('Hapiness score distribution across countries in 2021')
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    from scipy.stats import gaussian_kde

    # Create histogram data
    hist = np.histogram(whp_2021_report['Ladder score'], bins=20)
    bin_counts = hist[0]
    bin_edges = hist[1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate KDE for smooth curve
    kde = gaussian_kde(whp_2021_report['Ladder score'])
    x_range = np.linspace(whp_2021_report['Ladder score'].min(), 
                     whp_2021_report['Ladder score'].max(), 
                     200)
    kde_values = kde(x_range)

    # Create the figure
    fig = go.Figure()

    # Add histogram
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=bin_counts,
        name='Count',
        text=bin_counts,  # Add counts on top of bars
        textposition='outside',
        marker_color='lightblue',
        hovertemplate='Happiness Score: %{x:.2f}<br>Number of Countries: %{y}<extra></extra>'
    ))

    # Add KDE line
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_values * len(whp_2021_report) * (bin_edges[1] - bin_edges[0]),  # Scale KDE to match histogram height
        name='Density',
        line=dict(color='rgba(255, 0, 0, 0.6)', width=2),
        hovertemplate='Happiness Score: %{x:.2f}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Distribution of Happiness Scores Across Countries',
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95
        },
        xaxis_title='Happiness Score',
        yaxis_title='Number of Countries',
        plot_bgcolor='white',
        bargap=0.1,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=600
    )

    # Add gridlines
    fig.update_yaxes(
        gridcolor='lightgrey',
        gridwidth=0.5
    )
    fig.update_xaxes(
        gridcolor='lightgrey',
        gridwidth=0.5
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)