import logging
import os
import re
import swanlab

logger = logging.getLogger(__name__)

def sanitize_name(name):
    """清理名称，替换无效字符为下划线"""
    # 只保留允许的字符：0-9, a-z, A-Z, _, -, +, .
    return re.sub(r'[^0-9a-zA-Z_\-+.]', '_', name)

def init_swanlab_training(training_args):
    """
    初始化swanlab记录系统
    
    Args:
        training_args: 包含训练配置的参数对象
    """
    logger.info("正在初始化swanlab...")
    
    # 提取项目名和实验名，并清理无效字符
    raw_name = os.environ.get("SWANLAB_PROJECT", training_args.run_name)
    project_name = sanitize_name(raw_name)
    experiment_name = sanitize_name(training_args.run_name)
    
    if project_name != raw_name:
        logger.warning(f"项目名称已从 '{raw_name}' 清理为 '{project_name}' 以符合SwanLab命名规则")
    
    # 初始化swanlab
    swanlab.init(
        experiment_name=experiment_name,
        project=project_name,
        config=vars(training_args),  # 将训练参数作为配置记录
        dir=os.path.join(training_args.output_dir, "swanlab"),  # swanlab日志存储目录
    )
    
    logger.info(f"Swanlab已初始化：项目={project_name}，实验={experiment_name}")