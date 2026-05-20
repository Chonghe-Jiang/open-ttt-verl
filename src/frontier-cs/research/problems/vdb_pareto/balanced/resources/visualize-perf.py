# HNSW Faiss
# batch_time_vs_recall: [(133.60285758972168, np.float64(0.5462)), (57.738542556762695, np.float64(0.6948)), (56.23745918273926, np.float64(0.8175)), (75.2866268157959, np.float64(0.9012)), (146.10576629638672, np.float64(0.9543)), (288.12718391418457, np.float64(0.9779)), (486.65904998779297, np.float64(0.9882)), (969.1460132598877, np.float64(0.9902)), (1710.646629333496, np.float64(0.9912))]
# single_time_vs_recall: [(74.73087310791016, np.float64(0.5462)), (182.8896999359131, np.float64(0.6948)), (113.40975761413574, np.float64(0.8175)), (161.3025665283203, np.float64(0.9012)), (264.2638683319092, np.float64(0.9543)), (662.2898578643799, np.float64(0.9779)), (939.0251636505127, np.float64(0.9882)), (1273.374080657959, np.float64(0.9902)), (2269.6480751037598, np.float64(0.9912))]
# (ann-bench)
# HNSW paper implementation by GPT5
#  batch_time_vs_recall: [(10357.190370559692, np.float64(0.2459)), (10870.532274246216, np.float64(0.3443)), (23420.75228691101, np.float64(0.4445)), (75730.08227348328, np.float64(0.532)), (126461.98201179504, np.float64(0.6041)), (220550.12464523315, np.float64(0.6519))]
# single_time_vs_recall: [(796.410322189331, np.float64(0.2459)), (1011.8968486785889, np.float64(0.3443)), (5014.99080657959, np.float64(0.4445)), (7614.3224239349365, np.float64(0.532)), (13668.43056678772, np.float64(0.6041)), (8690.499782562256, np.float64(0.6519))]
# IVF Faiss
# batch_time_vs_recall: [(151.12853050231934, np.float64(0.4109)), (88.94872665405273, np.float64(0.525)), (74.97882843017578, np.float64(0.5911)), (94.30050849914551, np.float64(0.632)), (116.42789840698242, np.float64(0.7347)), (115.81301689147949, np.float64(0.8257)), (170.81069946289062, np.float64(0.8988)), (246.28925323486328, np.float64(0.9483)), (624.9041557312012, np.float64(0.9762)), (1144.5066928863525, np.float64(0.9877)), (3532.7444076538086, np.float64(0.9908))]
# single_time_vs_recall: [(1040.8475399017334, np.float64(0.4109)), (392.3466205596924, np.float64(0.525)), (424.790620803833, np.float64(0.5911)), (442.4548149108887, np.float64(0.632)), (431.40554428100586, np.float64(0.7347)), (546.5426445007324, np.float64(0.8257)), (711.9157314300537, np.float64(0.8988)), (890.9642696380615, np.float64(0.9483)), (1868.8640594482422, np.float64(0.9762)), (7412.7771854400635, np.float64(0.9877)), (16230.722904205322, np.float64(0.9908))]
import matplotlib.pyplot as plt


def main() -> None:
    # Times are assumed to be in milliseconds; tuples are (time, recall)
    batch_time_vs_recall = [
        (133.60285758972168, 0.5462),
        (57.738542556762695, 0.6948),
        (56.23745918273926, 0.8175),
        (75.2866268157959, 0.9012),
        (146.10576629638672, 0.9543),
        (288.12718391418457, 0.9779),
        (486.65904998779297, 0.9882),
        (969.1460132598877, 0.9902),
        (1710.646629333496, 0.9912),
    ]

    single_time_vs_recall = [
        (74.73087310791016, 0.5462),
        (182.8896999359131, 0.6948),
        (113.40975761413574, 0.8175),
        (161.3025665283203, 0.9012),
        (264.2638683319092, 0.9543),
        (662.2898578643799, 0.9779),
        (939.0251636505127, 0.9882),
        (1273.374080657959, 0.9902),
        (2269.6480751037598, 0.9912),
    ]

    # Sort by recall so curves are monotonic in x
    batch_time_vs_recall.sort(key=lambda t: t[1])
    single_time_vs_recall.sort(key=lambda t: t[1])

    batch_times = [t for t, r in batch_time_vs_recall]
    batch_recalls = [r for t, r in batch_time_vs_recall]
    single_times = [t for t, r in single_time_vs_recall]
    single_recalls = [r for t, r in single_time_vs_recall]

    plt.figure(figsize=(7, 4.5))
    plt.plot(batch_recalls, batch_times, marker="o", label="Batch")
    plt.plot(single_recalls, single_times, marker="s", label="Single")
    plt.xlabel("Recall")
    plt.ylabel("Time (ms)")
    plt.title("Recall vs Time")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("recall_vs_time.png", dpi=200)
    # For headless environments, we only save the figure.


if __name__ == "__main__":
    main()
