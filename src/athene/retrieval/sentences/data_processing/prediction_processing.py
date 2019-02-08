import json

from common.dataset.reader import JSONLineReader


def generate_prediction_files(predictions, p_sents_indexes, data_path, final_prediction_path):
    """
    transform the generated predictions from classifier to lists of dicts form to feed into the score system
    :param predictions:
    :param p_sents_indexes:
    :param data_path:
    :param final_prediction_path:
    :return:
    """
    jlr = JSONLineReader()

    final_predictions = []
    with open(data_path, "r") as f:
        lines = jlr.process(f)

        print(len(predictions))
        print(len(p_sents_indexes))
        print(len(lines))
        assert len(predictions) == len(p_sents_indexes) == len(lines)
        for idx, line in enumerate(lines):

            line['predicted_evidence'] = []
            line['predicted_label'] = 'refutes'
            predicted_sents = predictions[idx]
            sents_indexes = p_sents_indexes[idx]
            for i in range(len(sents_indexes)):
                if predicted_sents[i] == 1:
                    line['predicted_evidence'].append([sents_indexes[i][0], sents_indexes[i][1]])

            final_predictions.append(line)

    with open(final_prediction_path, "w") as f:

        for prediction in final_predictions:
            f.write(json.dumps(prediction) + '\n')

    return final_predictions
