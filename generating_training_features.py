import pandas as pd
import os
def _get_processed_df(filename) -> pd.DataFrame:
        """
        获取处理后的 DataFrame, 处理包括对 raw_df 排序, error_type 转独热编码, 填充缺失值, 添加奇偶校验特征
        :param sn_file: sn 文件名
        :return: 处理后的 DataFrame
        """
        parity_dict = dict()
        raw_df = pd.read_feather(filename)
        raw_df = raw_df.sort_values(by="LogTime").reset_index(drop=True)
        processed_df = raw_df[
            [
                "LogTime",
                "deviceID",
                "BankId",
                "RowId",
                "ColumnId",
                "MciAddr",
                "RetryRdErrLogParity",
            ]
        ].copy()
        processed_df["deviceID"] = (
            processed_df["deviceID"].fillna(-1).astype(int)
        )
        processed_df["error_type_is_READ_CE"] = (
            raw_df["error_type_full_name"] == "CE.READ"
        ).astype(int)
        processed_df["error_type_is_SCRUB_CE"] = (
            raw_df["error_type_full_name"] == "CE.SCRUB"
        ).astype(int)
        processed_df["CellId"] = (
            processed_df["RowId"].astype(str)
            + "_"
            + processed_df["ColumnId"].astype(str)
        )
        processed_df["position_and_parity"] = (
            processed_df["deviceID"].astype(str)
            + "_"
            + processed_df["BankId"].astype(str)
            + "_"
            + processed_df["RowId"].astype(str)
            + "_"
            + processed_df["ColumnId"].astype(str)
            + "_"
            + processed_df["RetryRdErrLogParity"].astype(str)
        )

        err_log_parity_array = (
            processed_df["RetryRdErrLogParity"]
            .fillna(0)
            .replace("", 0)
            .astype(np.int64)
            .values
        )
        bit_dq_burst_count = list()
        for idx, err_log_parity in enumerate(err_log_parity_array):
            if err_log_parity not in parity_dict:
                parity_dict[err_log_parity] = _get_bit_dq_burst_info(
                    err_log_parity
                )
            bit_dq_burst_count.append(parity_dict[err_log_parity])
        processed_df = processed_df.join(
            pd.DataFrame(
                bit_dq_burst_count,
                columns=[
                    "bit_count",
                    "dq_count",
                    "burst_count",
                    "max_dq_interval",
                    "max_burst_interval",
                ],
            )
        )
        return processed_df

import numpy as np
def _get_bit_dq_burst_info(parity: np.int64):
        """
        获取特定 parity 的奇偶校验信息
        :param parity: 奇偶校验值
        :return: parity 的奇偶校验信息
        bit_count: parity 错误 bit 数量
        dq_count: parity 错误 dq 数量
        burst_count: parity 错误 burst 数量
        max_dq_interval: parity 错误 dq 的最大间隔
        max_burst_interval: parity 错误 burst 的最大间隔
        """
        bin_parity = bin(parity)[2:].zfill(32)
        bit_count = bin_parity.count("1")
        binary_row_array = [bin_parity[i : i + 4].count("1") for i in range(0, 32, 4)]
        binary_row_array_indices = [
            idx for idx, value in enumerate(binary_row_array) if value > 0
        ]
        burst_count = len(binary_row_array_indices)
        max_burst_interval = (
            binary_row_array_indices[-1] - binary_row_array_indices[0]
            if binary_row_array_indices
            else 0
        )

        binary_column_array = [bin_parity[i::4].count("1") for i in range(4)]
        binary_column_array_indices = [
            idx for idx, value in enumerate(binary_column_array) if value > 0
        ]
        dq_count = len(binary_column_array_indices)
        max_dq_interval = (
            binary_column_array_indices[-1] - binary_column_array_indices[0]
            if binary_column_array_indices
            else 0
        )
        return bit_count, dq_count, burst_count, max_dq_interval, max_burst_interval

from dataclasses import dataclass, field
TIME_RELATED_LIST: list = field(
        default_factory=lambda: [15 * ONE_MINUTE, ONE_HOUR, 6 * ONE_HOUR], init=False
    )

def _calculate_ce_storm_count(
        log_times: np.ndarray,
        ce_storm_interval_seconds: int = 60,
        ce_storm_count_threshold: int = 10,
    ) -> int:
        """
        计算 CE 风暴的数量, CE 风暴定义:
        - 首先定义相邻 CE 日志: 若两个 CE 日志 LogTime 时间间隔 < 60s, 则为相邻日志;
        - 如果相邻日志的个数 >10, 则为发生 1 次 CE 风暴(注意: 如果相邻日志数量持续增长, 超过了 10, 则也只是记作 1 次 CE 风暴)

        :param log_times: 日志 LogTime 列表
        :param ce_storm_interval_seconds: CE 风暴的时间间隔阈值
        :param ce_storm_count_threshold: CE 风暴的数量阈值
        :return: CE风暴的数量
        """
        log_times = sorted(log_times)
        ce_storm_count = 0
        consecutive_count = 0

        for i in range(1, len(log_times)):
            if log_times[i] - log_times[i - 1] <= ce_storm_interval_seconds:
                consecutive_count += 1
            else:
                consecutive_count = 0
            if consecutive_count > ce_storm_count_threshold:
                ce_storm_count += 1
                consecutive_count = 0

        return ce_storm_count


def _get_temporal_features(
       window_df: pd.DataFrame, time_window_size: int
    ):
        """
        获取时间特征, 包括 CE 数量, 日志数量, CE 风暴数量, 日志发生频率等

        :param window_df: 时间窗口内的数据
        :param time_window_size: 时间窗口大小
        :return: 时间特征

        - read_ce_log_num, read_ce_count: 时间窗口内, 读 CE 的 count 总数, 日志总数
        - scrub_ce_log_num, scrub_ce_count: 时间窗口内, 巡检 CE 的 count 总数, 日志总数
        - all_ce_log_num, all_ce_count: 时间窗口内, 所有 CE 的 count 总数, 日志总数
        - log_happen_frequency: 日志发生频率
        - ce_storm_count: CE 风暴数量
        """
        error_type_is_READ_CE = window_df["error_type_is_READ_CE"].values
        error_type_is_SCRUB_CE = window_df["error_type_is_SCRUB_CE"].values
        ce_count = window_df["Count"].values

        temporal_features = dict()
        temporal_features["read_ce_log_num"] = error_type_is_READ_CE.sum()
        temporal_features["scrub_ce_log_num"] = error_type_is_SCRUB_CE.sum()
        temporal_features["all_ce_log_num"] = len(window_df)

        temporal_features["read_ce_count"] = (error_type_is_READ_CE * ce_count).sum()
        temporal_features["scrub_ce_count"] = (error_type_is_SCRUB_CE * ce_count).sum()
        temporal_features["all_ce_count"] = ce_count.sum()

        temporal_features["log_happen_frequency"] = (
            time_window_size / len(window_df) if not window_df.empty else 0
        )
        temporal_features["ce_storm_count"] = _calculate_ce_storm_count(
            window_df["LogTime"].values
        )
        return temporal_features

IMPUTE_VALUE = -1
def _unique_num_filtered(input_array: np.ndarray) -> int:
        """
        对输入的列表进行过滤, 去除值为 IMPUTE_VALUE 的元素后, 统计元素个数

        :param input_array: 输入的列表
        :return: 返回经过过滤后的列表元素个数
        """
        unique_array = np.unique(input_array)
        return len(unique_array) - int(IMPUTE_VALUE in unique_array)


def _get_spatio_features(window_df: pd.DataFrame):
        """
        获取空间特征, 包括故障模式, 同时发生行列故障的数量
        :param window_df: 时间窗口内的数据
        :return: 空间特征
        fault_mode_others: 其他故障, 即多个 device 发生故障
        fault_mode_device: device 故障, 相同 id 的 device 发生多个 bank 故障
        fault_mode_bank: bank 故障, 相同 id 的 bank 发生多个 row故障
        fault_mode_row: row 故障, 相同 id 的 row 发生多个不同 column 的 cell 故障
        fault_mode_column: column 故障, 相同 id 的 column 发生多个不同 row 的 cell 故障
        fault_mode_cell: cell 故障, 发生多个相同 id 的 cell 故障
        fault_row_num: 同时发生 row 故障的行个数
        fault_column_num: 同时发生 column 故障的列个数
        """
        spatio_features = {
            "fault_mode_others": 0,
            "fault_mode_device": 0,
            "fault_mode_bank": 0,
            "fault_mode_row": 0,
            "fault_mode_column": 0,
            "fault_mode_cell": 0,
            "fault_row_num": 0,
            "fault_column_num": 0,
        }

        if _unique_num_filtered(window_df["deviceID"].values) > 1:
            spatio_features["fault_mode_others"] = 1
        elif _unique_num_filtered(window_df["BankId"].values) > 1:
            spatio_features["fault_mode_device"] = 1
        elif (
            _unique_num_filtered(window_df["ColumnId"].values) > 1
            and _unique_num_filtered(window_df["RowId"].values) > 1
        ):
            spatio_features["fault_mode_bank"] = 1
        elif _unique_num_filtered(window_df["ColumnId"].values) > 1:
            spatio_features["fault_mode_row"] = 1
        elif _unique_num_filtered(window_df["RowId"].values) > 1:
            spatio_features["fault_mode_column"] = 1
        elif _unique_num_filtered(window_df["CellId"].values) == 1:
            spatio_features["fault_mode_cell"] = 1

        # 记录相同行对应的列地址信息
        row_pos_dict = {}
        # 记录相同列对应的行地址信息
        col_pos_dict = {}

        for device_id, bank_id, row_id, column_id in zip(
            window_df["deviceID"].values,
            window_df["BankId"].values,
            window_df["RowId"].values,
            window_df["ColumnId"].values,
        ):
            current_row = "_".join([str(pos) for pos in [device_id, bank_id, row_id]])
            current_col = "_".join(
                [str(pos) for pos in [device_id, bank_id, column_id]]
            )
            row_pos_dict.setdefault(current_row, [])
            col_pos_dict.setdefault(current_col, [])
            row_pos_dict[current_row].append(column_id)
            col_pos_dict[current_col].append(row_id)

        for row in row_pos_dict:
            if _unique_num_filtered(np.array(row_pos_dict[row])) > 1:
                spatio_features["fault_row_num"] += 1
        for col in col_pos_dict:
            if _unique_num_filtered(np.array(col_pos_dict[col])) > 1:
                spatio_features["fault_column_num"] += 1

        return spatio_features

def _get_err_parity_features(window_df: pd.DataFrame):
        """
        获取奇偶校验特征
        :param window_df: 时间窗口内的数据
        :return: 奇偶校验特征
        error_bit_count: 时间窗口内, 总错误 bit 数
        error_dq_count: 时间窗口内, 总 dq 错误数
        error_burst_count: 时间窗口内, 总 burst 错误数
        max_dq_interval: 时间窗口内, 每个 parity 最大错误 dq 距离的最大值
        max_burst_interval: 时间窗口内, 每个 parity 最大错误 burst 距离的最大值
        dq_count=n: dq错误数=n 的总数量, n 取值 [1,2,3,4],默认值: 0
        burst_count=n: burst错误数=n 的总数量, n 取值 [1,2,3,4,5,6,7,8],默认值: 0
        """
        err_parity_features = dict()

        err_parity_features["error_bit_count"] = window_df["bit_count"].values.sum()
        err_parity_features["error_dq_count"] = window_df["dq_count"].values.sum()
        err_parity_features["error_burst_count"] = window_df["burst_count"].values.sum()
        err_parity_features["max_dq_interval"] = window_df[
            "max_dq_interval"
        ].values.max()
        err_parity_features["max_burst_interval"] = window_df[
            "max_burst_interval"
        ].values.max()

        dq_counts = dict()
        burst_counts = dict()
        for i in zip(window_df["dq_count"].values, window_df["burst_count"].values):
            dq_counts[i[0]] = dq_counts.get(i[0], 0) + 1
            burst_counts[i[1]] = burst_counts.get(i[1], 0) + 1
        DQ_COUNT = 4
        # 计算 'dq错误数=n' 的总数量, DDR4 内存的 DQ_COUNT 为 4, 因此 n 取值 [1,2,3,4]
        for dq in range(1, DQ_COUNT + 1):
            err_parity_features[f"dq_count={dq}"] = dq_counts.get(dq, 0)
        # 计算 'burst错误数=n' 的总数量, DDR4 内存的 BURST_COUNT 为 8, 因此 n 取值 [1,2,3,4,5,6,7,8]
        for burst in [1, 2, 3, 4, 5, 6, 7, 8]:
            err_parity_features[f"burst_count={burst}"] = burst_counts.get(burst, 0)

        return err_parity_features

from dataclasses import dataclass, field
TIME_RELATED_LIST: list = field(
        default_factory=lambda: [15 * ONE_MINUTE, ONE_HOUR, 6 * ONE_HOUR], init=False
    )

ONE_MINUTE = 60
feature_interval: int = 15 * ONE_MINUTE ##每15分钟一个时间窗口，每个时间窗口一条数据
ONE_HOUR = 3600
ONE_DAY = 86400
TIME_RELATED_LIST = [15 * ONE_MINUTE, ONE_HOUR, 6 * ONE_HOUR]
    
TIME_WINDOW_SIZE_MAP =  {
            15 * ONE_MINUTE: "15m",
            ONE_HOUR: "1h",
            6 * ONE_HOUR: "6h",
        }



###所有数据的
def process_single_sn(sn_file):
        """
        处理单个 sn 文件, 获取不同尺度的时间窗口特征
        :param sn_file: sn 文件名
        """
        new_df = _get_processed_df(sn_file)
        new_df["time_index"] = new_df["LogTime"] // feature_interval
        log_times = new_df["LogTime"].values

        window_end_times = new_df.groupby("time_index")["LogTime"].max().values
        window_start_times = window_end_times - 6 * ONE_HOUR

        start_indices = np.searchsorted(log_times, window_start_times, side="left")
        end_indices = np.searchsorted(log_times, window_end_times, side="right")

        combined_dict_list = []
        for start_idx, end_idx, end_time in zip(
            start_indices, end_indices, window_end_times
        ):
            combined_dict = {}
            window_df = new_df.iloc[start_idx:end_idx]
            combined_dict["LogTime"] = window_df["LogTime"].values.max()
            window_df = window_df.assign(
                Count=window_df.groupby("position_and_parity")[
                    "position_and_parity"
                ].transform("count")
            )
            window_df = window_df.drop_duplicates(
                subset="position_and_parity", keep="first"
            )
            log_times = window_df["LogTime"].values
            end_logtime_of_filtered_window_df = window_df["LogTime"].values.max()

            for time_window_size in TIME_RELATED_LIST:
                index = np.searchsorted(
                    log_times,
                    end_logtime_of_filtered_window_df - time_window_size,
                    side="left",
                )
                window_df_copy = window_df.iloc[index:]

                temporal_features = _get_temporal_features(
                    window_df_copy, time_window_size
                )
                spatio_features = _get_spatio_features(window_df_copy)
                err_parity_features = _get_err_parity_features(window_df_copy)

                combined_dict.update(
                    {
                        f"{key}_{TIME_WINDOW_SIZE_MAP[time_window_size]}": value
                        for d in [
                            temporal_features,
                            spatio_features,
                            err_parity_features,
                        ]
                        for key, value in d.items()
                    }
                )
            combined_dict_list.append(combined_dict)
        combined_df = pd.DataFrame(combined_dict_list)
        return combined_df

if __name__ == "__main__":
    filepathin = "./select_type_A/"
    filepathout = "./select_type_A_feature/"
    all_items = os.listdir(filepathin)
    # 筛选出文件名称
    file_names = [item for item in all_items if os.path.isfile(os.path.join(filepathin, item))]
    for value in file_names:
        file_name_now = filepathin + value
        df = process_single_sn(file_name_now)
        file_name_out = filepathout + value
        df.to_csv(file_name_out, index=False)
    filepathin = "./select_type_B/"
    filepathout = "./select_type_B_feature/"
    all_items = os.listdir(filepathin)
    # 筛选出文件名称
    file_names = [item for item in all_items if os.path.isfile(os.path.join(filepathin, item))]
    for value in file_names:
        file_name_now = filepathin + value
        df = process_single_sn(file_name_now)
        file_name_out = filepathout + value
        df.to_csv(file_name_out, index=False)