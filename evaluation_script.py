import mysql.connector
import matplotlib.pyplot as plt
import itertools

mydb = mysql.connector.connect(
    host="isys-db.cs.uni-paderborn.de",
    user="pgotfml",
    passwd="XXX",
    database="XXX",
    ssl_ca="",
    ssl_key=None,
    ssl_cert=None
)

mycursor = mydb.cursor()

# get all configurations
mycursor.execute(
    "SELECT DISTINCT sampling_strategy FROM active_learning_curves ")
sampling_strategies = [x[0] for x in mycursor.fetchall()]

mycursor.execute("SELECT DISTINCT train_ratio FROM active_learning_curves ")
train_ratios = [x[0] for x in mycursor.fetchall()]

mycursor.execute(
    "SELECT DISTINCT old_data_in_minibatch_ratio FROM active_learning_curves ")
old_data_ratios = [x[0] for x in mycursor.fetchall()]

mycursor.execute("SELECT DISTINCT minibatch_size FROM active_learning_curves ")
minibatch_sizes = [x[0] for x in mycursor.fetchall()]

mycursor.execute("SELECT DISTINCT dataset FROM active_learning_curves ")
datasets = [x[0] for x in mycursor.fetchall()]

mycursor.execute(
    "SELECT DISTINCT remove_queried_dyads FROM active_learning_curves ")
remove_queried_dyads = [x[0] for x in mycursor.fetchall()]

mycursor.execute("SELECT DISTINCT measure FROM active_learning_curves ")
measures = [x[0] for x in mycursor.fetchall()]

mycursor.execute("SELECT DISTINCT seed FROM active_learning_curves ")
seeds = [int(x[0]) for x in mycursor.fetchall()]

print(sampling_strategies)
print(train_ratios)
print(old_data_ratios)
print(minibatch_sizes)
print(datasets)
print(remove_queried_dyads)
print(measures)
print(seeds)

for config in itertools.product(sampling_strategies, train_ratios, old_data_ratios, minibatch_sizes, datasets, remove_queried_dyads, measures, seeds):
    query = "SELECT query_step, score FROM active_learning_curves WHERE sampling_strategy = %s AND train_ratio = %s AND old_data_in_minibatch_ratio = %s AND minibatch_size = %s AND dataset = %s AND remove_queried_dyads = %s AND measure = %s AND seed = %s"
    mycursor.execute(query, config[0:8])
    results = mycursor.fetchall()
    print(config)
    print(results)
    query_steps = [x[0] for x in results]
    scores = [x[1] for x in results]
    print(query_steps)
    print(scores)
    plt.plot(query_steps, scores)
    plt.show()



mycursor.close()
