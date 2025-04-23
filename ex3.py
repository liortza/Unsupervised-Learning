# Adva Cohen 323840561, Lior Tzahar 208629808
import pickle
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns

CLUSTERS_NUM = 9
RARE_WORDS = []
TOPICS = ['acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']


def init_documents_info(documents):
    docs_dict = {}
    document_number = 1
    for doc in documents:
        docs_dict[document_number] = {}
        docs_dict[document_number]['ntk'] = {}
        for word in doc.split():
            if word not in RARE_WORDS:
                docs_dict[document_number]['ntk'][word] = docs_dict[document_number]['ntk'][word] + 1 if word in \
                                                                                                         docs_dict[
                                                                                                             document_number][
                                                                                                             'ntk'] else 1
        cluster_num = ((document_number - 1) % CLUSTERS_NUM) + 1
        docs_dict[document_number]['wti'] = {key: 0 for key in range(1, 10)}
        docs_dict[document_number]['wti'][cluster_num] = 1
        doc_unrare_words = [word for word in doc.split() if word not in RARE_WORDS]
        docs_dict[document_number]['nt'] = len(doc_unrare_words)
        document_number += 1
    return docs_dict


def calc_alpha_i(cluster_num, docs_dict):
    n = len(docs_dict)
    sum = 0
    for doc_num in docs_dict:
        wti = docs_dict[doc_num]['wti'][cluster_num]
        sum += wti
    return sum / n


def calc_pik(cluster_num, docs_dict, lamda=1.05):
    cluster_dict = {}
    numerator_cache = {}
    dominator_cache = {}
    voc = set()

    # Build the vocabulary set from all documents
    for doc_num in docs_dict:
        voc.update(docs_dict[doc_num]['ntk'].keys())
    voc_size = len(voc)
    # Precompute numerator and dominator for each document
    for doc_num, doc_data in docs_dict.items():
        nt = doc_data['nt']
        wti = doc_data['wti'][cluster_num]
        for word in voc:
            ntk = docs_dict[doc_num]['ntk'][word] if word in docs_dict[doc_num]['ntk'] else 0
            if word not in numerator_cache:
                numerator_cache[word],dominator_cache[word] = 0, 0
            numerator_cache[word] += wti * ntk
            dominator_cache[word] += wti * nt

    # Compute cluster probabilities for each word
    for word in voc:
        numerator = numerator_cache.get(word, 0)
        dominator = dominator_cache.get(word, 0)
        cluster_dict[word] = (numerator + lamda) / (dominator + (voc_size * lamda))
    return cluster_dict


def calc_zi(alphas, pik_dict, docs_dict):
    zi = {}  # doc-> [zi for each cluster i]
    for i in range(1, 10):
        for doc_num in docs_dict:
            sum_k = 0
            for word in docs_dict[doc_num]['ntk']:
                ntk = docs_dict[doc_num]['ntk'][word]
                pik = pik_dict[i][word]
                sum_k += ntk * math.log(pik)
            if doc_num not in zi:
                zi[doc_num] = {i: math.log(alphas[i]) + sum_k}
            else:
                zi[doc_num][i] = math.log(alphas[i]) + sum_k
    return zi


def calc_log_likelihood(docs_dict, zi_dict, k=10):
    total_log_likelihood = 0
    for doc_num in docs_dict:
        # Compute m^t (max z_i for document t)
        m = max(zi_dict[doc_num].values())
        # Compute the sum of e^{z_i - m}
        sum_k = 0
        for i in range(1, 10):
            zi = zi_dict[doc_num][i]
            if zi - m >= -k:  # Ignore small values for numerical stability
                sum_k += math.exp(zi - m)
        # Check if sum_k is valid to avoid math errors
        if sum_k > 0:
            total_log_likelihood += m + math.log(sum_k)
        else:
            raise ValueError(f"Invalid sum_k value (<= 0) for document {doc_num}. Check z_i values.")
    return total_log_likelihood


def calc_wti(zi_dict, docs_dict, k=10):
    for doc_num in zi_dict:
        m = max(zi_dict[doc_num].values())
        for i in range(1, 10):
            zi = zi_dict[doc_num][i]
            if m > zi + k:
                docs_dict[doc_num]['wti'][i] = 0
            else:
                numerator = math.exp(zi - m)
                d = sum([math.exp(zi_dict[doc_num][j] - m)
                         for j in range(1, 10) if zi_dict[doc_num][j] - m >= -k])
                docs_dict[doc_num]['wti'][i] = (numerator / d)


def em_algo(alphas, docs_dict, pik, epsilon=0.01):
    ll = -math.inf
    new_ll = ll
    iteration_number = 0
    log_likelihood_values = []
    while new_ll >= ll:  # num of iterations
        ll = new_ll
        for i in alphas:
            if not alphas[i]:
                alphas[i] = epsilon
        zi = calc_zi(alphas, pik, docs_dict)
        calc_wti(zi, docs_dict)
        for i in range(1, 10):
            alphas[i] = calc_alpha_i(i, docs_dict)
        pik = {i: calc_pik(i, docs_dict) for i in range(1, 10)}
        new_ll = calc_log_likelihood(docs_dict, zi)
        iteration_number += 1
        log_likelihood_values.append(new_ll)
        print(f'log likelihood = {new_ll}')
    print(f"number of iterations to convergence: {iteration_number}")
    return list(zip(range(1, iteration_number), log_likelihood_values))


def initialization(file):
    documents, topics = [], []
    text_lines = file.readlines()
    count = 0
    for i in range(2, len(text_lines), 4):  # Every 4th line starting from index 2
        documents.append(text_lines[i])
        topics.append(text_lines[i - 2][:-1].rstrip('>').split()[2:])
        count += 1
    return documents, topics


def get_rare_words(documents):
    words_count = {}
    for doc in documents:
        words = doc.split()
        for word in words:
            words_count[word] = words_count.get(word, 0) + 1
    rare_words = [word for word in words_count if words_count[word] <= 3]
    return rare_words


def create_graph(data, x_label, y_label):
    # Separate the data into x and y values
    x, y = zip(*data)
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b', label=y_label)
    # Customize the plot
    plt.title(f'{x_label} vs. {y_label}', fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    # Show the plot
    plt.show()


def calculate_perlexity_values(log_likelihood_list, docs_dict):
    perlexity_values = []
    nts = sum([doc_data['nt'] for doc_num, doc_data in docs_dict.items()])
    for iteration, log_likelihood in log_likelihood_list:
        perlexity_values.append((iteration, np.exp(-log_likelihood / nts)))
    return perlexity_values


def create_confusion_matrix(docs_dict, topics):
    matrix = np.zeros((9, 10))
    topic_index = dict(list(zip(TOPICS, range(CLUSTERS_NUM))))

    for doc in docs_dict:
        cluster = max(docs_dict[doc]['wti'], key=docs_dict[doc]['wti'].get) - 1
        documents_topics = topics[doc - 1]
        for topic in documents_topics:
            matrix[cluster][topic_index[topic]] += 1
            matrix[cluster][CLUSTERS_NUM] += 1
    return matrix


def create_matrix_gui(matrix):
    # Create a random matrix
    rows = CLUSTERS_NUM
    cols = len(TOPICS)
    # Create labels for rows and columns
    row_labels = [f'Cluster {i + 1}' for i in range(rows)]
    col_labels = TOPICS + ['Total']
    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="pink", xticklabels=col_labels, yticklabels=row_labels, cbar=True)
    # Title and display
    plt.title('Confusion Matrix')
    plt.show()


def create_clusters_histograms(matrix):
    matrix = np.delete(matrix, 9, axis=1)
    for i in range(CLUSTERS_NUM):
        row = matrix[i, :]
        plt.bar(TOPICS, row, color='blue', edgecolor='black', alpha=0.7)
        plt.title(f"Histogram for Cluster {i + 1}")
        plt.xticks(ticks=np.arange(len(TOPICS)), labels=TOPICS)
        plt.show()


def calc_accuracy(docs_dict, topics, matrix):
    correct, total = 0, 0
    matrix = np.delete(matrix, 9, axis=1)
    for doc in docs_dict:
        doc_cluster = max(docs_dict[doc]['wti'], key=docs_dict[doc]['wti'].get) - 1
        doc_topic = topics[(doc - 1)]
        cluster_topic = np.where(matrix[doc_cluster] == max(matrix[doc_cluster]))[0][0]
        topic = TOPICS[cluster_topic]
        if topic in doc_topic:
            correct += 1
        total += 1
    print(f'Accuracy: {correct / total}')

def fast_init():
    """
    run the report code with results from pre-run of the EM algorithm - for faster execution
    """
    with open("develop.txt", 'r') as develop_file:
        documents, topics = initialization(develop_file)
    with open("documents.pkl", 'rb') as f:
        docs_dict = pickle.load(f)
    with open("logliklihood.pkl", 'rb') as f:
        log_likelihood_list = pickle.load(f)
    for ll in log_likelihood_list:
        print(f'Log Likelihood for iteration {ll[0]}: {ll[1]}')
    return log_likelihood_list, docs_dict, topics


def full_init():
    """
    run the EM algorithm and calculate the log likelihood values, takes about 5 minutes
    """
    global RARE_WORDS
    with open("develop.txt", 'r') as develop_file:
        documents, topics = initialization(develop_file)
    RARE_WORDS = get_rare_words(documents)
    alphas = {}
    docs_dict = init_documents_info(documents)
    for i in range(1, 10):
        alphas[i] = calc_alpha_i(i, docs_dict)
    pik = {i: calc_pik(i, docs_dict) for i in range(1, 10)}
    log_likelihood_list = em_algo(alphas, docs_dict, pik)
    with open("documents.pkl", 'wb') as f:
        pickle.dump(docs_dict, f)
    with open("logliklihood.pkl", 'wb') as f:
        pickle.dump(log_likelihood_list, f)
    return log_likelihood_list, docs_dict, topics

def main():
    log_likelihood_list, docs_dict, topics = fast_init()
    # log_likelihood_list, docs_dict, topics = full_init()

    # Report
    create_graph(log_likelihood_list, 'Number of Iterations', 'Log-Likelihood')
    preplexity_values = calculate_perlexity_values(log_likelihood_list, docs_dict)
    create_graph(preplexity_values, 'Number of Iterations', 'Perplexity')
    confusion_matrix = create_confusion_matrix(docs_dict, topics)
    create_matrix_gui(confusion_matrix)
    create_clusters_histograms(matrix=confusion_matrix)
    calc_accuracy(docs_dict, topics, confusion_matrix)


if __name__ == '__main__':
    main()  # Run the main function when the script is executed
