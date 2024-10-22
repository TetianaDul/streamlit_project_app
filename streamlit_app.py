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

    st.header("Data Visualization")

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
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)


    #histogram of hapiness score distribution over the years
    st.subheader('Hapiness score distribution across countries over the years')

    import plotly.graph_objects as go
    import numpy as np
    from scipy.stats import gaussian_kde
    import streamlit as st

    # Get unique years for the slider
    years = sorted(df['year'].unique())

    # Create year selector
    selected_year = st.slider('Select Year', 
                         min_value=int(min(years)), 
                         max_value=int(max(years)), 
                         value=int(max(years)))  # Default to most recent year


    # Filter data for selected year
    year_data = df[df['year'] == selected_year]['Ladder score']

    # Create histogram data
    hist = np.histogram(year_data, bins=20)
    bin_counts = hist[0]
    bin_edges = hist[1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate KDE for smooth curve
    kde = gaussian_kde(year_data)
    x_range = np.linspace(year_data.min(), year_data.max(), 200)
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
        marker_color='blue',
        hovertemplate='Happiness Score: %{x:.2f}<br>Number of Countries: %{y}<extra></extra>'
    ))

    # Add KDE line
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_values * len(year_data) * (bin_edges[1] - bin_edges[0]),  # Scale KDE to match histogram height
        name='Density',
        line=dict(color='rgba(255, 0, 0, 0.6)', width=2),
        hovertemplate='Happiness Score: %{x:.2f}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': f'Distribution of Happiness Scores Across Countries ({selected_year})',
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
        gridwidth=0.5,
        zeroline=True
    )
    fig.update_xaxes(
        gridcolor='lightgrey',
        gridwidth=0.5,
        zeroline=True
    )

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)


    #add a correlation heatmap over the years
    st.subheader('Correlation of different factors to hapiness over the years')

    import plotly.graph_objects as go
    import numpy as np
    import streamlit as st

    # Get unique years for the slider
    years = sorted(df['year'].unique())

    # Create year selector with a unique key
    selected_year = st.slider('Select Year', 
                            min_value=int(min(years)), 
                            max_value=int(max(years)), 
                            value=int(max(years)),
                            key='correlation_year_slider')  # Added unique key

    factors = ['Ladder score', 'Logged GDP per capita', 'Social support', 
          'Healthy life expectancy', 'Freedom to make life choices', 
          'Generosity', 'Perceptions of corruption']

    # Filter data for selected year and calculate correlation matrix
    year_data = df[df['year'] == selected_year][factors]
    corr_matrix = year_data.corr()

    # Create the heatmap
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=corr_matrix,
        x=factors,
        y=factors,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        hoverongaps=False,
        hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': f'Correlation Matrix of Happiness Factors ({selected_year})',
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95
        },
        width=800,
        height=800,
        xaxis={
            'tickangle': -45,
            'side': 'bottom'
        },
        yaxis={
            'tickangle': 0,
        }
    )


    # Find strongest positive and negative correlations
    correlations = []
    for i in range(len(factors)):
        for j in range(i+1, len(factors)):
            correlations.append({
                'factor1': factors[i],
                'factor2': factors[j],
                'correlation': corr_matrix.iloc[i, j]
            })

    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

    # Display the plot
    st.plotly_chart(fig, use_container_width=True, key='correlation_matrix')



    #add a world map with the ladder score
    st.subheader('Hapiness over the years on a world map')

    import streamlit as st
    import plotly.express as px

    # Create the visualization
    df_sorted = df.sort_values(by='year')

    fig = px.choropleth(
        df_sorted,
        locations='Country name',
        locationmode='country names',
        color='Ladder score',
        hover_name='Country name',
        animation_frame='year',
        color_continuous_scale=px.colors.sequential.Plasma,
        title="World Happiness Ladder Score by Country Over Time"
    )
    # Make the figure bigger by updating layout
    fig.update_layout( 
        height=700,
        title={
            'text': "World Happiness Ladder Score by Country Over Time",
            'y':0.95,  
            'x':0.5,   
            'xanchor': 'center',  
            'yanchor': 'top'  
        },
        margin=dict(t=80, l=20, r=20, b=20),  # Minimized margins
        )

    # Display the plot with expanded width
    st.plotly_chart(fig, use_container_width=True)


    #add factor contribution to happiness in 2021
    st.subheader('Contribution of different factors to happiness in 2021')

    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go

    # Selecting relevant columns
    contribution_cols = [
        'Explained by: Log GDP per capita',
        'Explained by: Social support',
        'Explained by: Healthy life expectancy',
        'Explained by: Freedom to make life choices',
        'Explained by: Generosity',
        'Explained by: Perceptions of corruption'
    ]

    # Sort by Ladder score
    whp_2021_report_sorted = whp_2021_report.sort_values(by='Ladder score', ascending=False)

    # Select top 10 and bottom 10 countries
    top_10 = whp_2021_report_sorted.head(10)
    bottom_10 = whp_2021_report_sorted.tail(10)

    # Combine top n and bottom n into one DataFrame
    whp_2021_report_filtered = pd.concat([top_10, bottom_10])

    # Create plotting data
    fig = go.Figure()

    # Add each contribution as a separate bar
    for col in contribution_cols:
        fig.add_trace(go.Bar(
            name=col.replace('Explained by: ', ''),
            x=whp_2021_report_filtered['Country name'],
            y=whp_2021_report_filtered[col],
        ))

    # Update layout
    fig.update_layout(
        barmode='stack',
        title=f'Contribution of Factors to Ladder Score for Top 10 and Bottom 10 Countries',
        xaxis_title='Country',
        yaxis_title='Contribution to Ladder Score',
        xaxis_tickangle=45,
        height=800,
        legend_title='Contributing Factors',
        showlegend=True
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


#On the modelling page (page [3])
if page == pages[3]:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import GridSearchCV


    # Load your datasets
    X_train = pd.read_csv('X_train_encoded.csv')
    y_train = pd.read_csv('y_train.csv')
    X_test = pd.read_csv('X_test_encoded.csv')
    y_test = pd.read_csv('y_test.csv')

    # Title of the app
    st.header('Machine Learning Models')

    # Add the objective
    st.subheader("Objective")
    st.write("The goal of this analysis was to identify the best performing machine learning model for predicting world happiness levels in every country using various machine learning techniques.")

    # Add an expander for model selection
    with st.expander("Models Chosen", expanded=False):
        st.write("""
        We evaluated and compared the performance of six models to determine which model best captures the patterns in the data and provides accurate predictions on unseen data:
        - Linear Regression
        - Random Forest
        - Lasso
        - Decision Tree Regressor
        """)

        st.subheader("Execution of Models")
        st.write("""
        - Model instantiation
        - Training the model on the full training set X_train and y_train (80%, 20%)
        - Making predictions on the test set X_test and y_test
        - Evaluating the performance of the models using appropriate metrics
        - Interpreting the coefficients to understand the impact of each feature on the target variable
        - Visualizing and analyzing the results.
        """)

    # Add a new expander for machine learning models
    with st.expander("Machine Learning Models", expanded=False):
        st.subheader("Metric Comparison Chart")
    
        # Creating a DataFrame for metrics
        metrics_data = {
            "Model": [
                "Linear Regression", 
                "Random Forest Regressor", 
                "Lasso", 
                "Decision Tree Regressor"
            ],
            "R² Score (train)": [
                "74.6%", 
                "85.1%", 
                "62.2%", 
                "90%"
            ],
            "R² Score (test)": [
                "73.9%", 
                "80.5%", 
                "62.9%", 
                "75%"
            ],
            "MAE": [
                "0.450", 
                "0.389", 
                "0.569", 
                "0.411"
            ],
            "MSE": [
                "0.337", 
                "0.251", 
                "0.480", 
                "0.321"
            ],
            "RMSE": [
                "0.580", 
                "0.501", 
                "0.693", 
                "0.321"
            ],
            "Important Features": [
                "GDP, Social support, Life expectancy", 
                "GDP, Life expectancy, Social support", 
                "GDP, Life expectancy", 
                "GDP, Life expectancy, Social support"
            ]
        }

        metrics_df = pd.DataFrame(metrics_data)
        st.table(metrics_df)  # Display the DataFrame as a table

        # Key insights
        st.subheader("Key Insights")
        st.write("""
        - **Decision Tree Regressor**: High training R² (90%), lower test R² (75%) – signs of overfitting  
        - **Random Forest Regressor**: Consistent performance, 85.1% (train) and 80.5% (test) – accurate for unseen data  
        - **Lasso**: Lowest R² on test set (62.9%) – unreliable for predicting happiness  
        - **Linear Regression**: Similar R² (~74%) for both training and test – lower than Random Forest
        """)

    # Add a new expander for feature importance visualization
    with st.expander("Importance Bar Visualization", expanded=False):
        # A. Linear Regression Feature Importance
        st.subheader("A. Linear Regression Feature Importance")

        # Apply standardization using StandardScaler
        sc = StandardScaler()
        num = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']

        X_train[num] = sc.fit_transform(X_train[num])
        X_test[num] = sc.transform(X_test[num])

        # Instantiate a LinearRegression model and train it on the training data
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # Get the feature names and coefficients
        feature_names = X_train.columns
        coefficients = regressor.coef_.flatten() # Flatten coefficients to 1D if needed

        # Create a DataFrame for easy plotting
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': coefficients})

        # Sort the DataFrame by absolute importance
        importance_df['Absolute Importance'] = np.abs(importance_df['Importance'])
        importance_df = importance_df.sort_values(by='Absolute Importance', ascending=False)

        # Plotting the bar chart in Streamlit
        fig, ax = plt.subplots(figsize=(10, 6)) # Create a figure with the specified size
        ax.barh(importance_df['Feature'], importance_df['Absolute Importance'], color='green')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance for Linear Regression')
        ax.invert_yaxis()  # To display the most important feature at the top

        # Use Streamlit's pyplot method to display the plot
        st.pyplot(fig)

    
        # B. Random Forest Feature Importance
        st.subheader("B. Random Forest Feature Importance")

        # Train the Random Forest model
        model_forest_opt = RandomForestRegressor(max_depth=5, random_state=42)
        model_forest_opt.fit(X_train, y_train.values.ravel())  # Fit the model

        # Extract feature importances
        importances = model_forest_opt.feature_importances_

        # Create a DataFrame for feature names and their importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        # Sort the DataFrame by importance
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Plot the feature importances for Random Forest
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')

        # Add labels and title
        plt.title('Feature Importance in Random Forest Model')
        plt.xlabel('Importance')
        plt.ylabel('Feature')

        # Display the plot in the Streamlit app
        st.pyplot(plt)

        # C. Lasso Feature Importance
        st.subheader("C. Lasso Feature Importance")

        # Impute missing values using the median for the training set
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)  # Fit on X_train
        X_test_imputed = imputer.transform(X_test)  # Apply the same transformation to X_test

        # Initialize and fit the Lasso model
        lasso = Lasso(alpha=0.1)  # Adjust the alpha as necessary
        lasso.fit(X_train_imputed, y_train)

        # Get the coefficients for Lasso
        coefficients = pd.Series(lasso.coef_, index=X_train.columns)

        # Plotting Lasso feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x=coefficients.values, y=coefficients.index)
        plt.title('Feature Importance (Lasso Coefficients)')
        plt.xlabel('Importance')

        # Display the plot in the Streamlit app
        st.pyplot(plt)

        # D. Decision Tree Regressor Feature Importance
        st.subheader("D. Decision Tree Regressor Feature Importance")

        # Define a range of max_depth values to test
        param_grid = {'max_depth': list(range(1, 21))}  # Testing depths from 1 to 20

        # Initialize the Decision Tree Regressor
        dt_regressor = DecisionTreeRegressor(random_state=42)

        # Set up GridSearch with cross-validation
        grid_search = GridSearchCV(
            estimator=dt_regressor,
            param_grid=param_grid,
            cv=5,  # 5-fold cross-validation
            scoring='r2',
            n_jobs=-1
        )

        # Fit GridSearch to find the best max_depth
        grid_search.fit(X_train, y_train.values.ravel())

        # Extract the best max_depth
        best_max_depth = grid_search.best_params_['max_depth']
        st.write(f"Optimal max_depth: {best_max_depth}")

        # Train the Decision Tree Regressor with the optimal max_depth
        model = DecisionTreeRegressor(max_depth=best_max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions on the training and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Feature importance plot
        feature_importances = model.feature_importances_

        # Barplot of feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances, y=feature_names, palette='viridis')
        plt.title("Feature Importance in Decision Tree Regressor")
        plt.xlabel("Importance")
        plt.ylabel("Features")

        # Display the plot in the Streamlit app
        st.pyplot(plt)
	        # Insights at the bottom of the box
        st.subheader("Insights")
        st.write("""
        - Key factors for most models: GDP, Social Support, and Life Expectancy  
        - Lasso: Penalizes most factors, giving mostly importance to GDP
        - Random Forest, Decision Tree: Focus more on GDP and Life Expectancy, suggesting a prediction of general, reliable trends  
        """)

    # Add a new expander for scatterplot visualization
    with st.expander("Scatterplot for Ladder Score Visualization", expanded=False):
        # A. Linear Regression
        st.subheader("A. Linear Regression")

        # Display the scatter plot between predicted and true values
        pred_test = regressor.predict(X_test)
    
        # Create the scatter plot for Linear Regression
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(pred_test, y_test, c='green', label='Predicted vs True')
    
        # Add the y = x line
        plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color='red', label='y = x')
    
        plt.xlabel("Predicted Values")
        plt.ylabel("True Values")
        plt.title('Linear Regression for Ladder Score')
        plt.legend()
    
        # Display the plot in the Streamlit app
        st.pyplot(fig)

        # B. Random Forest
        st.subheader("B. Random Forest")

        # Display the scatter plot between predicted and true values
        pred_test_rf = model_forest_opt.predict(X_test)

        # Create the scatter plot for Random Forest
        fig_rf = plt.figure(figsize=(10, 10))
        plt.scatter(pred_test_rf, y_test, c='green', label='Predicted vs True')

        # Add the y = x line
        plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color='red', label='y = x')

        plt.xlabel("Predicted Values")
        plt.ylabel("True Values")
        plt.title('Random Forest for Ladder Score')
        plt.legend()

        # Display the plot in the Streamlit app
        st.pyplot(fig_rf)

        # C. Lasso
        st.subheader("C. Lasso")

        # Make predictions using the Lasso model
        y_test_pred = lasso.predict(X_test_imputed)

        # Scatter plot for Lasso predictions
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Ideal line
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values (Test Set - Lasso)')
    
        # Display the plot in the Streamlit app
        st.pyplot(plt)

        # D. Decision Tree Regressor
        st.subheader("D. Decision Tree Regressor")

        # Make predictions using the Decision Tree model
        y_test_pred_dt = model.predict(X_test)

        # Scatter plot for Decision Tree predictions
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_test_pred_dt, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Ideal line
        plt.xlabel('Test Values')
        plt.ylabel('Predicted Values')
        plt.title('Test vs Predicted Values (Decision Tree)')
    
        # Display the plot in the Streamlit app
        st.pyplot(plt)

        # Key Insights at the bottom of the box
        st.subheader("Key Insights")
        st.write("""
        - Even though they all perform well, it is clear Lasso has the most difficulties.
        - They all present outliers. Other factors have an impact that the models can't predict.
        - Random Forest has the best results.
        """)


#On the prediction page (page [4])
if page == pages[4]:
    
    st.title('Hello, Streamlit! Whats UUUUP')
    
    from sklearn.ensemble import RandomForestRegressor

    X_train = pd.read_csv('X_train_encoded.csv')
    y_train = pd.read_csv('y_train.csv')
    X_test = pd.read_csv('X_test_encoded.csv')
    y_test = pd.read_csv('y_test.csv')

    # Dropping all features except of the most important ones
    X_train.drop(['Country_encoded','Generosity','Perceptions of corruption','year'],axis=1,inplace=True)
    X_test.drop(['Country_encoded','Generosity','Perceptions of corruption','year'],axis=1,inplace=True)

    # Model fit and predictions
    RFR=RandomForestRegressor(max_depth=5, random_state=42)
    RFR.fit(X_train,y_train)
    predictions=RFR.predict(X_test)

    # Showing the most important feature importances
    feat_df=pd.DataFrame(RFR.feature_importances_,index=X_train.columns,columns=['Values'])
    st.subheader("Feature Importances")
    # Body text
    st.text("We chose the RFR model, because it has the best score compared \n to the others. The mot important feature importances are displayed in the \n following graph")
    sorted_feat_df = feat_df.sort_values(by='Values',ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=sorted_feat_df.index[:4], y=sorted_feat_df['Values'][:4])
    plt.title('Feature Importances of the model')
    plt.ylabel('Degree of Importance')
    plt.xlabel('Importances')
    plt.show()
    st.pyplot(plt)

    gdp_per_capita = st.slider("GDP per Capita", 0.0, 10.0, 9.4)
    social_support = st.slider("Social Support", 0.0, 1.0, 0.8)
    healthy_life_expectancy = st.slider("Healthy Life Expectancy", 0.0, 100.0, 63.8)
    freedom_to_make_choices = st.slider("Freedom to Make Life Choices", 0.0, 1.0, 0.7)

    # Create a DataFrame with the input values
    user_input = pd.DataFrame({'Logged GDP per capita': [gdp_per_capita],
        'Social support': [social_support],
        'Healthy life expectancy': [healthy_life_expectancy],
        'Freedom to make life choices': [freedom_to_make_choices]})
    # Make prediction based on user input
    prediction = RFR.predict(user_input)
    st.subheader("Predicted Life Ladder Score:")
    st.write("Prediction:", prediction[0])