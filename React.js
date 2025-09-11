import React, { useState, useEffect } from 'react';
import { AlertTriangle, Shield, TrendingUp, DollarSign, Users, Activity, Eye, RefreshCw, Filter, Download } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, Area, AreaChart } from 'recharts';

const FraudDetectionDashboard = () => {
  // State management
  const [dashboardData, setDashboardData] = useState({
    total_transactions: 15420,
    fraud_transactions: 312,
    fraud_rate_percent: 2.02,
    total_amount: 2847650.75,
    fraud_amount: 45680.20,
    prevented_fraud_amount: 125000.00,
    roi_percent: 284.5,
    savings: 115000.00
  });

  const [trends, setTrends] = useState([
    { date: '2024-01-01', total_transactions: 450, fraud_transactions: 12, fraud_rate: 2.67, total_amount: 87500, fraud_amount: 2340 },
    { date: '2024-01-02', total_transactions: 520, fraud_transactions: 8, fraud_rate: 1.54, total_amount: 95600, fraud_amount: 1680 },
    { date: '2024-01-03', total_transactions: 480, fraud_transactions: 15, fraud_rate: 3.13, total_amount: 92300, fraud_amount: 3750 },
    { date: '2024-01-04', total_transactions: 610, fraud_transactions: 11, fraud_rate: 1.80, total_amount: 118500, fraud_amount: 2890 },
    { date: '2024-01-05', total_transactions: 580, fraud_transactions: 18, fraud_rate: 3.10, total_amount: 105400, fraud_amount: 4120 },
    { date: '2024-01-06', total_transactions: 490, fraud_transactions: 9, fraud_rate: 1.84, total_amount: 89200, fraud_amount: 1950 },
    { date: '2024-01-07', total_transactions: 630, fraud_transactions: 22, fraud_rate: 3.49, total_amount: 125600, fraud_amount: 5680 }
  ]);

  const [cohorts, setCohorts] = useState([
    { cohort_id: 'geo_NY', total_transactions: 3420, fraud_rate: 2.8, avg_transaction_amount: 185.50, prevention_savings: 28500 },
    { cohort_id: 'geo_CA', total_transactions: 2890, fraud_rate: 1.9, avg_transaction_amount: 198.75, prevention_savings: 31200 },
    { cohort_id: 'geo_TX', total_transactions: 2150, fraud_rate: 3.2, avg_transaction_amount: 142.30, prevention_savings: 18900 },
    { cohort_id: 'payment_credit_card', total_transactions: 8920, fraud_rate: 2.4, avg_transaction_amount: 176.80, prevention_savings: 65400 },
    { cohort_id: 'payment_debit_card', total_transactions: 4250, fraud_rate: 1.6, avg_transaction_amount: 154.20, prevention_savings: 28700 },
    { cohort_id: 'amount_high', total_transactions: 1250, fraud_rate: 4.8, avg_transaction_amount: 850.40, prevention_savings: 45200 }
  ]);

  const [alerts, setAlerts] = useState([
    { id: 1, transaction_id: 'TXN-8941', alert_type: 'High Risk', risk_score: 0.89, timestamp: '2024-01-07T14:30:00', status: 'active' },
    { id: 2, transaction_id: 'TXN-8942', alert_type: 'Unusual Pattern', risk_score: 0.76, timestamp: '2024-01-07T14:25:00', status: 'active' },
    { id: 3, transaction_id: 'TXN-8943', alert_type: 'Velocity Alert', risk_score: 0.82, timestamp: '2024-01-07T14:20:00', status: 'investigating' },
    { id: 4, transaction_id: 'TXN-8944', alert_type: 'Device Anomaly', risk_score: 0.71, timestamp: '2024-01-07T14:15:00', status: 'active' }
  ]);

  const [selectedTimeRange, setSelectedTimeRange] = useState('30d');
  const [selectedCohortType, setSelectedCohortType] = useState('geographic');
  const [loading, setLoading] = useState(false);

  // Simulated API calls
  const refreshData = () => {
    setLoading(true);
    // Simulate API delay
    setTimeout(() => {
      setLoading(false);
    }, 1000);
  };

  // Chart colors
  const COLORS = {
    primary: '#3B82F6',
    success: '#10B981',
    danger: '#EF4444',
    warning: '#F59E0B',
    info: '#6366F1',
    dark: '#1F2937'
  };

  const pieColors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4'];

  // Format currency
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  // Format percentage
  const formatPercentage = (value) => {
    return `${value.toFixed(2)}%`;
  };

  // Get risk level color
  const getRiskColor = (score) => {
    if (score >= 0.8) return 'text-red-600 bg-red-100';
    if (score >= 0.5) return 'text-yellow-600 bg-yellow-100';
    return 'text-green-600 bg-green-100';
  };

  // Metric Card Component
  const MetricCard = ({ title, value, subValue, icon: Icon, trend, color = 'blue' }) => (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200 hover:shadow-xl transition-all duration-300">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <p className={`text-3xl font-bold text-${color}-600 mb-1`}>{value}</p>
          {subValue && <p className="text-sm text-gray-500">{subValue}</p>}
          {trend && (
            <div className={`flex items-center mt-2 text-sm ${trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
              <TrendingUp className={`w-4 h-4 mr-1 ${trend < 0 ? 'rotate-180' : ''}`} />
              {Math.abs(trend)}% vs last period
            </div>
          )}
        </div>
        <div className={`p-3 bg-${color}-100 rounded-full`}>
          <Icon className={`w-8 h-8 text-${color}-600`} />
        </div>
      </div>
    </div>
  );

  // Alert Component
  const AlertItem = ({ alert }) => (
    <div className="flex items-center justify-between p-4 bg-white rounded-lg border border-gray-200 hover:shadow-md transition-shadow">
      <div className="flex items-center space-x-3">
        <div className={`p-2 rounded-full ${getRiskColor(alert.risk_score)}`}>
          <AlertTriangle className="w-4 h-4" />
        </div>
        <div>
          <p className="font-medium text-gray-900">{alert.transaction_id}</p>
          <p className="text-sm text-gray-600">{alert.alert_type}</p>
          <p className="text-xs text-gray-500">{new Date(alert.timestamp).toLocaleString()}</p>
        </div>
      </div>
      <div className="text-right">
        <div className={`px-3 py-1 rounded-full text-xs font-medium ${getRiskColor(alert.risk_score)}`}>
          Risk: {(alert.risk_score * 100).toFixed(0)}%
        </div>
        <div className={`mt-1 px-2 py-1 rounded text-xs ${
          alert.status === 'active' ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800'
        }`}>
          {alert.status}
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-gray-900 mb-2">
                AI Fraud Detection Dashboard
              </h1>
              <p className="text-gray-600">Real-time fraud monitoring and risk analysis</p>
            </div>
            <div className="flex space-x-3">
              <select 
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={selectedTimeRange}
                onChange={(e) => setSelectedTimeRange(e.target.value)}
              >
                <option value="7d">Last 7 days</option>
                <option value="30d">Last 30 days</option>
                <option value="90d">Last 90 days</option>
              </select>
              <button 
                onClick={refreshData}
                disabled={loading}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </button>
            </div>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <MetricCard
            title="Total Transactions"
            value={dashboardData.total_transactions.toLocaleString()}
            subValue={`${formatCurrency(dashboardData.total_amount)} total volume`}
            icon={Activity}
            trend={12.5}
            color="blue"
          />
          <MetricCard
            title="Fraud Rate"
            value={formatPercentage(dashboardData.fraud_rate_percent)}
            subValue={`${dashboardData.fraud_transactions} fraud cases`}
            icon={AlertTriangle}
            trend={-8.2}
            color="red"
          />
          <MetricCard
            title="Prevented Loss"
            value={formatCurrency(dashboardData.prevented_fraud_amount)}
            subValue="AI Prevention System"
            icon={Shield}
            trend={15.8}
            color="green"
          />
          <MetricCard
            title="ROI"
            value={formatPercentage(dashboardData.roi_percent)}
            subValue={`${formatCurrency(dashboardData.savings)} net savings`}
            icon={TrendingUp}
            trend={22.1}
            color="purple"
          />
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Fraud Trends */}
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-gray-900">Fraud Detection Trends</h3>
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
                  Total Transactions
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                  Fraud Rate
                </div>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={trends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleDateString()}
                  formatter={(value, name) => [
                    name === 'fraud_rate' ? `${value}%` : value,
                    name === 'fraud_rate' ? 'Fraud Rate' : 'Total Transactions'
                  ]}
                />
                <Area
                  yAxisId="left"
                  type="monotone"
                  dataKey="total_transactions"
                  stackId="1"
                  stroke={COLORS.primary}
                  fill={COLORS.primary}
                  fillOpacity={0.6}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="fraud_rate"
                  stroke={COLORS.danger}
                  strokeWidth={3}
                  dot={{ fill: COLORS.danger, strokeWidth: 2, r: 4 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Risk Distribution */}
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <h3 className="text-xl font-semibold text-gray-900 mb-6">Risk Level Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={[
                    { name: 'Low Risk (0-0.3)', value: 78, color: COLORS.success },
                    { name: 'Medium Risk (0.3-0.7)', value: 18, color: COLORS.warning },
                    { name: 'High Risk (0.7-1.0)', value: 4, color: COLORS.danger }
                  ]}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {[
                    { name: 'Low Risk', value: 78, color: COLORS.success },
                    { name: 'Medium Risk', value: 18, color: COLORS.warning },
                    { name: 'High Risk', value: 4, color: COLORS.danger }
                  ].map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Cohort Analysis & Active Alerts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Cohort Analysis */}
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-gray-900">Cohort Analysis</h3>
              <select 
                className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500"
                value={selectedCohortType}
                onChange={(e) => setSelectedCohortType(e.target.value)}
              >
                <option value="geographic">Geographic</option>
                <option value="behavioral">Behavioral</option>
                <option value="temporal">Temporal</option>
              </select>
            </div>
            <div className="space-y-3">
              {cohorts.slice(0, 6).map((cohort, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <p className="font-medium text-gray-900">{cohort.cohort_id.replace('_', ' ').toUpperCase()}</p>
                    <p className="text-sm text-gray-600">
                      {cohort.total_transactions.toLocaleString()} transactions
                    </p>
                  </div>
                  <div className="text-right">
                    <p className={`text-sm font-medium ${
                      cohort.fraud_rate > 3 ? 'text-red-600' : 
                      cohort.fraud_rate > 2 ? 'text-yellow-600' : 'text-green-600'
                    }`}>
                      {cohort.fraud_rate.toFixed(1)}% fraud rate
                    </p>
                    <p className="text-xs text-gray-500">
                      {formatCurrency(cohort.prevention_savings)} saved
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Active Alerts */}
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-gray-900">Active Alerts</h3>
              <div className="flex items-center space-x-2">
                <div className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm font-medium">
                  {alerts.filter(a => a.status === 'active').length} Active
                </div>
                <button className="p-2 text-gray-400 hover:text-gray-600 transition-colors">
                  <Eye className="w-4 h-4" />
                </button>
              </div>
            </div>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {alerts.map((alert) => (
                <AlertItem key={alert.id} alert={alert} />
              ))}
            </div>
            <div className="mt-4 pt-4 border-t border-gray-200">
              <button className="w-full text-center text-sm text-blue-600 hover:text-blue-800 font-medium transition-colors">
                View All Alerts →
              </button>
            </div>
          </div>
        </div>

        {/* Transaction Volume by Hour */}
        <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200 mb-8">
          <h3 className="text-xl font-semibold text-gray-900 mb-6">Transaction Volume by Hour</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart
              data={Array.from({ length: 24 }, (_, i) => ({
                hour: i,
                transactions: Math.floor(Math.random() * 500) + 100,
                fraud: Math.floor(Math.random() * 20) + 1
              }))}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" tickFormatter={(value) => `${value}:00`} />
              <YAxis />
              <Tooltip 
                labelFormatter={(value) => `Hour: ${value}:00`}
                formatter={(value, name) => [value, name === 'fraud' ? 'Fraud Cases' : 'Total Transactions']}
              />
              <Bar dataKey="transactions" fill={COLORS.primary} />
              <Bar dataKey="fraud" fill={COLORS.danger} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-center space-x-4">
          <button className="flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            <Download className="w-5 h-5 mr-2" />
            Export Report
          </button>
          <button className="flex items-center px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors">
            <Filter className="w-5 h-5 mr-2" />
            Advanced Filters
          </button>
          <button className="flex items-center px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors">
            <Users className="w-5 h-5 mr-2" />
            Manage Users
          </button>
        </div>
      </div>
    </div>
  );
};

export default FraudDetectionDashboard;
