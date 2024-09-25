# Internship-projects
The project aims to analyze the historical data of Attrition and determine the main causes for high attrition in the company. Utilized Python language by using Jupyter as an IDE.

## Libraries:
Within Python, made use of **Numpy, Pandas** libraries for data manipulation and analysis along with visualization tools of **Seaborn, Matplotlib and Plotly**.

## Working on the dataset:
### EDA(Exploratory Data Analysis)
Initially performed a basic understanding of the dataset, columns, its values and also, the statistical analysis using the info() and describe() methods respectively.

Post this, confirmed there were multiple columns such as WorkLifeBalance, EnvironmentSatisfaction, JobSatisfaction, etc., that had missing values. Using the _displot_ viz understood the distribution of data for these columns.
From these plots, handled the data by filling in the mean or mode according the basic statistical imputation handling techniques.

Also, dropped certain columns: 'Over18', 'StandardHours', 'EmployeeID' because these are constant for all employees and had no deviation which did not impact the final Attrition value or the dataset as a whole.

_This improved the runtime of the DataFrame significantly_

# Analysis and Visualizations:
After the steps of cleaning and preprocessing, started the analysis to understand or gain insights on why the attrition is high, factors influencing it and also, wanted to create a model that could predict the instance of Attrition based on all important signals.

### Insights from the analysis:
1. First inference from the data indicated a dsiparity in the salary ranges of the departments and this caused HR department to be the highest contributor for Attrition.
   ![1](https://github.com/user-attachments/assets/cb55f862-b750-4df3-9699-71cd2028ff3d)

   ![2](https://github.com/user-attachments/assets/9b599c6d-4d4b-4f67-b81b-70d857d80b63)
2. Males had a higher number of attrition.

   ![3](https://github.com/user-attachments/assets/1fde2fff-00f8-404b-9951-d4fcd4ee0ca5) 
3. New employees i.e., employees with tenure less than 1 year have left the company compared to more tenured ones.

   ![4](https://github.com/user-attachments/assets/76a570ed-3573-430e-a942-051019a7d22a)
   
4. Also, for tenured folks, the count leaving saw a sharper decline after 5 and 10 years in the company.
* 0-5: 20.87%
* 5-10: 11.8%
* 10-15: 10.96%
5. New employees have a lower monthly income, job satisfaction and environment satisfaction ratings that show a direct impact on the decision to leave the company.

### **Recommendations:**
1. The New hires need to have a better induction that can connect them to the Company's vision and mission and also, have the salary ranges on par with the market value.
2. The starting phase or years seem to be very problematic seeing the highest attrition percentage and the factors of Environment Satisfaction and Job Satisfaction show that there are major issues at the beginning.
3. Focussing specifically on New Hires, there is again HR department that has a negative impact due to Salary ranges. Also, New Hires with no prior experience or just 1 year of total experience also, seem to have varied salaries causing varied ratings in Job Satisfaction. Having a more streamlined or fixed ranges of salary based on years of experience will help improve the satisfaction and reduce attrition among New Employees
4. HR department: Increase the Hiring count, improve the range of salaries provided and also, make the salary proportional to the years of experience which would increase the upper Quartile values.
Sales Department: Salaries are within a good range and has proportionalities as well but need to increase the hiring numbers as well to reduce the impact on Attrition.
5. The Work environment seems passive or not engaging which added to the reason of high attrition within the first year of joining. This again showed an impact along with indicators such as 'Job Involvement', 'Work-Life Balance', 'Job Satisfaction' as well.
6. Post the employees reaching average tenure of 20-25 years there is a sharp decline in most of the above indicatos as well. Even with Performance Rating being good, there are less indications of improvement so, providing opportunities/resources for upskilling, promotion, etc., can enhance employee satisfaction during these middle years.
